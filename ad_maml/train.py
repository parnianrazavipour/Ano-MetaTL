import torch
from torch import nn, optim
import learn2learn as l2l

from ad_maml.evaluate import eval_model, calculate_loss


def train_model(model, embedder, train_sampler, val_dataset, test_dataset, args):
    best_val_auc = 0
    device = args.device
    loss_func = nn.BCELoss()
    maml = l2l.algorithms.MetaSGD(model, lr=args.adapt_lr)
    optimizer = optim.Adam(maml.parameters(), args.meta_lr)

    for iter in range(args.epoch):
        meta_train_loss = 0
        maml.train()
        batch = train_sampler.next_batch()
        for item in batch:
            learner = maml.clone()

            support_x = torch.tensor(item[1]).to(device)
            support_y = torch.tensor(item[2], dtype=torch.float32).to(device)

            support_loss, _ = calculate_loss(support_x, support_y, learner, embedder, loss_func)
            learner.adapt(support_loss)

            query_norm_x = torch.tensor(item[3]).to(device)
            query_norm_y = torch.tensor(0.0).to(device)
            query_norm_loss, _ = calculate_loss(query_norm_x, query_norm_y, learner, embedder, loss_func)

            query_ano_x = torch.tensor(item[4]).to(device)
            query_ano_y = torch.tensor(1.0).to(device)
            query_ano_loss, _ = calculate_loss(query_ano_x, query_ano_y, learner, embedder, loss_func)

            meta_train_loss += (query_norm_loss + query_ano_loss) / 2
        meta_train_loss = meta_train_loss / len(batch)
        print(f'Iteration: {iter}, Meta Train Loss: {meta_train_loss.item()}')

        optimizer.zero_grad()
        meta_train_loss.backward()
        optimizer.step()

        if iter % args.eval_epoch == 0:
            print('Validating:')
            meta_val_loss, val_auc = eval_model(maml, embedder, val_dataset, loss_func, device)
            print(f'Iteration: {iter}, Meta Validation Loss: {meta_val_loss}, AUROC: {val_auc}')

            print('Testing:')
            meta_test_loss, test_auc = eval_model(maml, embedder, test_dataset, loss_func, device)
            print(f'Iteration: {iter}, Meta Test Loss: {meta_test_loss}, AUROC: {test_auc}')

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(maml.state_dict(), args.model_save_path)
                print(f'New model saved. Validation AUROC: {val_auc}, Test AUROC: {test_auc}')
