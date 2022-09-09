import torch
from sklearn.metrics import roc_auc_score


def calculate_loss(data, label, learner, embedder, loss_func):
    data_embed = embedder(data)
    data_ano = data_embed[-1].unsqueeze(0)
    data_embed = data_embed.reshape(-1).unsqueeze(0)
    preds = learner(data_embed, data_ano)
    preds = preds.squeeze(0)
    label = label.unsqueeze(0)
    loss = loss_func(preds, label)
    return loss, preds


def calculate_auroc(norm_preds, ano_preds):
    norm_preds = torch.tensor(norm_preds)
    ano_preds = torch.tensor(ano_preds)
    all_preds = torch.concat([norm_preds, ano_preds])
    pred_labels = [0] * len(norm_preds) + [1] * len(ano_preds)
    return roc_auc_score(pred_labels, all_preds)


def eval_model(maml, embedder, test_dataset, loss_func, device):
    meta_test_loss = 0
    all_query_norm_preds = []
    all_query_ano_preds = []
    maml.eval()
    for item in test_dataset:
        learner = maml.clone()

        support_x = torch.tensor(item[1]).to(device)
        support_y = torch.tensor(item[2], dtype=torch.float32).to(device)

        support_loss, _ = calculate_loss(support_x, support_y, learner, embedder, loss_func)
        learner.adapt(support_loss)

        query_norm_x = torch.tensor(item[3]).to(device)
        query_norm_y = torch.tensor(0.0).to(device)
        query_norm_loss, query_norm_preds = calculate_loss(query_norm_x, query_norm_y, learner, embedder, loss_func)
        all_query_norm_preds.append(query_norm_preds.detach())

        query_ano_x = torch.tensor(item[4]).to(device)
        query_ano_y = torch.tensor(1.0).to(device)
        query_ano_loss, query_ano_preds = calculate_loss(query_ano_x, query_ano_y, learner, embedder, loss_func)
        all_query_ano_preds.append(query_ano_preds.detach())

        meta_test_loss += (query_norm_loss.detach() + query_ano_loss.detach()) / 2
    meta_test_loss = meta_test_loss / len(test_dataset)
    auc = calculate_auroc(all_query_norm_preds, all_query_ano_preds)

    return meta_test_loss.item(), auc
