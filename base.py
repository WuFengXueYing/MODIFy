import os
import time
import copy

from sklearn.preprocessing import MinMaxScaler

import numpy as np

import torch
from matplotlib import pyplot as plt

from sklearn.manifold import TSNE

from torch import nn
import logging

from model import MainModel
from sklearn.metrics import ndcg_score


class BaseModel(nn.Module):
    def __init__(self, event_num, metric_num, node_num, device, lr=1e-3, epoches=50, beta = 0.1, patience=5, result_dir='./',
                 hash_id=None, **kwargs):
        super(BaseModel, self).__init__()

        self.epoches = epoches
        self.lr = lr
        self.beta = beta
        self.patience = patience  # > 0: use early stop
        self.device = device

        self.model_save_dir = os.path.join(result_dir, hash_id)

        self.model = MainModel(event_num, metric_num, node_num, device, **kwargs)
        self.model.to(device)

        logging.info(
            "Model Parameters: hash_id={}, data={}, device={}, lr={}, epoches={}".format(
                hash_id, kwargs.get("data"),  device, lr, epoches
            ))


    def evaluate(self, test_loader, datatype="Test"):
        self.model.eval()
        hrs, ndcgs = np.zeros(5), np.zeros(5)
        TN, TP, FP, FN = 0, 0, 0,0
        batch_cnt, epoch_loss = 0, 0.0
        inference_times = []
        visualize_embedding, visualize_label = [], []
        with torch.no_grad():
            for graph, ground_truths in test_loader:

                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                start_time = time.time()

                res = self.model.forward(graph.to(self.device), ground_truths)


                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()


                batch_size = graph.batch_size
                inference_time = (end_time - start_time) / batch_size
                inference_times.append(inference_time)


                embeddings = self.model.encoder(graph.to(self.device))  # [batch_size, feat_out_dim]


                visualize_embedding.append(embeddings.cpu().numpy())

                visualize_label.extend(ground_truths.numpy())


                for idx, faulty_nodes in enumerate(res["y_pred"]):
                    culprit = ground_truths[idx].item()
                    if culprit == -1:
                        if faulty_nodes[0] == -1:
                            TN += 1
                        else:
                            FP += 1
                    else:
                        if faulty_nodes[0] == -1:
                            FN += 1
                        else:
                            TP += 1
                            rank = list(faulty_nodes).index(culprit)
                            for j in range(5):
                                hrs[j] += int(rank <= j)
                                ndcgs[j] += ndcg_score([res["y_prob"][idx]], [res["pred_prob"][idx]], k=j + 1)
                epoch_loss += res["loss"].item()
                batch_cnt += 1
            epoch_loss = epoch_loss / batch_cnt
            logging.info(" testing loss: {:.5f} ".format(epoch_loss))

        visualize_embedding = np.vstack(visualize_embedding)
        visualize_label = np.array(visualize_label)


        avg_time = sum(inference_times) / len(inference_times)
        logging.info(f"{datatype} -- Average Inference Time: {avg_time:.6f}s per sample")


        pos = TP + FN
        eval_results = {
            "F1": TP * 2.0 / (TP + FP + pos) if (TP + FP + pos) > 0 else 0,
            "Rec": TP * 1.0 / pos if pos > 0 else 0,
            "Pre": TP * 1.0 / (TP + FP) if (TP + FP) > 0 else 0
        }

        for j in [1, 3, 5]:
            eval_results["HR@" + str(j)] = hrs[j - 1] * 1.0 / pos
            eval_results["ndcg@" + str(j)] = ndcgs[j - 1] * 1.0 / pos

        logging.info(
            "{} -- {}".format(datatype, ", ".join([k + ": " + str(f"{v:.4f}") for k, v in eval_results.items()])))

        return eval_results, visualize_embedding, visualize_label



    def fit(self, train_loader, test_loader=None, evaluation_epoch=10):


        best_hr1, coverage, best_state, eval_res = -1, None, None, None

        pre_loss, worse_count = float("inf"), 0


        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.99)


        for epoch in range(1, self.epoches + 1):
            self.model.train()
            batch_cnt, epoch_loss = 0, 0.0
            epoch_time_start = time.time()
            for graph, label in train_loader:

                optimizer.zero_grad()

                loss = self.model.forward(graph.to(self.device), label)['loss']

                loss.backward()
                # if self.debug:
                #     for name, parms in self.model.named_parameters():
                #         if name=='encoder.graph_model.net.weight':
                #             print(name, "--> grad:",parms.grad)
                optimizer.step()
                epoch_loss += loss.item()
                batch_cnt += 1



            epoch_time_elapsed = time.time() - epoch_time_start
            epoch_loss = epoch_loss / batch_cnt
            logging.info("Epoch {}/{}, training loss: {:.5f} [{:.2f}s]".format(epoch, self.epoches, epoch_loss,
                                                                               epoch_time_elapsed))


            if epoch_loss > pre_loss:
                worse_count += 1
                if self.patience > 0 and worse_count >= self.patience:
                    logging.info("Early stop at epoch: {}".format(epoch))
                    break
            else:
                worse_count = 0
            pre_loss = epoch_loss

            ####### Evaluate test data during training #######
            if (epoch + 1) % evaluation_epoch == 0:

                test_results, embedding, labels = self.evaluate(test_loader, datatype="Test")
                perplexity = 30
                tsne = TSNE(n_components=2, random_state=10, perplexity=perplexity,learning_rate='auto', n_iter=2000)
                tsne_result = tsne.fit_transform(embedding)
                self.visualize(tsne_result, epoch, labels, perplexity)

                if test_results["HR@1"] > best_hr1:
                    best_hr1, eval_res, coverage = test_results["HR@1"], test_results, epoch
                    best_state = copy.deepcopy(self.model.state_dict())
                self.save_model(best_state)

        if coverage > 5:
            logging.info("* Best result got at epoch {} with HR@1: {:.4f}".format(coverage, best_hr1))
        else:
            logging.info("Unable to convergence!")

        return eval_res, coverage

    def visualize(self, tsne_result, epoch, labels, perplexity):
        save_dir = f"./Fig/visualization/perplexity_{perplexity}"
        os.makedirs(save_dir, exist_ok=True)
        scaler = MinMaxScaler(feature_range=(0, 1))
        tsne_result = scaler.fit_transform(tsne_result)
        service_names = {
            0: 'compose-post-service',
            1: 'home-timeline-service',
            2: 'media-service',
            3: 'nginx-web-server',
            4: 'post-storage-service',
            5: 'social-graph-service',
            6: 'text-service',
            7: 'unique-id-service',
            8: 'url-shorten-service',
            9: 'user-mention-service',
            10: 'user-service',
            11: 'user-timeline-service'
        }
        valid_mask = labels != -1
        tsne_anomalies = tsne_result[valid_mask]
        fault_nodes = labels[valid_mask]

        plt.figure(figsize=(12, 10))
        unique_nodes = np.unique(fault_nodes)
        colors = plt.cm.get_cmap('tab20', len(unique_nodes))

        for i, node in enumerate(unique_nodes):
            idxs = fault_nodes == node
            plt.scatter(tsne_anomalies[idxs, 0],
                        tsne_anomalies[idxs, 1],
                        c=np.array(colors(i)).reshape(1,-1),
                        label=service_names.get(node, f"Node {node}"),
                        alpha=0.7,
                        edgecolors='white')

        plt.legend()


        if epoch is not None:
            filename = f"{save_dir}/tsne_epoch_{epoch}.pdf"
        else:
            filename = f"{save_dir}/tsne_final.pdf"

        plt.savefig(filename, format="pdf", bbox_inches="tight", dpi=300)
        plt.close()

    def load_model(self, model_save_file=""):
        self.model.load_state_dict(torch.load(model_save_file, map_location=self.device))

    def save_model(self, state, file=None):
        if file is None: file = os.path.join(self.model_save_dir, "model.ckpt")
        try:
            torch.save(state, file, _use_new_zipfile_serialization=False)
        except:
            torch.save(state, file)
