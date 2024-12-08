import pytorch_lightning as pl
import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.nn import functional as F
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from sklearn.metrics import recall_score, precision_score, confusion_matrix, precision_recall_curve
import wandb
from torch.utils.data import WeightedRandomSampler
import matplotlib.pyplot as plt

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, output_attn_w=False, n_hidden=64, dropout=0.1):
        """
        Args:
          embed_dim: Number of input and output features.
          n_heads: Number of attention heads in the Multi-Head Attention.
          n_hidden: Number of hidden units in the Feedforward (MLP) block.
          dropout: Dropout rate after the first layer of the MLP and in two places on the main path (before
                   combining the main path with a skip connection).
        """
        super(TransformerBlock,self).__init__()
        self.mh = nn.MultiheadAttention(embed_dim,n_heads,dropout=dropout)
        self.drop1= nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.drop2= nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim,n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden,embed_dim))
        self.output_attn_w = output_attn_w

    def forward(self, x):
        """
        Args:
          x of shape (max_seq_length, batch_size, embed_dim): Input sequences.
          
        Returns:
          xm of shape (max_seq_length, batch_size, embed_dim): Encoded input sequences.
          attn_w of shape (batch_size, max_seq_length, max_seq_length)
        """
        xm, attn_w= self.mh(x,x,x)
        xm = self.drop1(xm)
        xm = self.norm1(x+xm)
        x = self.ff(xm)
        x = self.drop2(x)
        xm = self.norm2(x+xm)
        return (xm,attn_w) if self.output_attn_w else xm
    

class Classifier(nn.Module): 
    # input_dim: 129024
    # hidden_dim: 64
    def __init__(self, input_dim, hidden_dim, dropout_linear):
        super(Classifier, self).__init__()
        self.relu = nn.ReLU()
        self.downsampling_linear = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_linear)
        self.res_block1 = ResidualBlock(hidden_dim, dropout_linear)
        self.final_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.downsampling_linear(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.res_block1(x)
        x = self.final_layer(x)
        return x
    

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_linear):
        super(ResidualBlock, self).__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_linear)
        self.bn2 = nn.BatchNorm1d(hidden_dim)  
        self.linear2 = nn.Linear(hidden_dim, hidden_dim) 

    def forward(self, x):
        residual = x
        out = self.bn1(x) 
        out = self.relu(out)
        out = self.linear1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)  
        out = self.linear2(out)
        out += residual 
        return out


class BetaVanillaModel(pl.LightningModule):
    def __init__(self, embed_dim, max_seq_length, device_, trbV_embed_len, trbJ_embed_len, mhc_embed_len, hyperparameters):
        super(BetaVanillaModel, self).__init__()
        """
        This model uses beta only information!
        Furthermore, this model DOES NOT (!) use physicochemical properties.        
        """
        self.save_hyperparameters()
        # for evaluation of the model
        self.auroc = BinaryAUROC(thresholds=None)
        self.avg_precision = BinaryAveragePrecision(thresholds=None)
        self.hyperparameters = hyperparameters
        self.device_ = device_
        self.max_seq_length = max_seq_length

        self.test_predictions = []
        self.test_labels = []
        self.test_tasks = []
        self.sources = []

        self.epitopes = []
        self.trb_cdr3s = []
        
        self.allele_info_dim = 1024
        self.trbV_embed = nn.Embedding(trbV_embed_len, self.allele_info_dim)
        self.trbJ_embed = nn.Embedding(trbJ_embed_len, self.allele_info_dim)
        self.mhc_embed = nn.Embedding(mhc_embed_len, self.allele_info_dim)
        
        # Define TransformerBlock for TRA+Epitope and TRB+Epitope after physico attention
        # Note: embed_dim must be dividable by num_heads!
        self.transformer_in = embed_dim
        self.num_heads = 4
        # same as in EPIC-TRACE paper
        self.n_hidden = int(1.5*self.transformer_in)
        self.multihead_attn_global = TransformerBlock(self.transformer_in, self.num_heads, False, self.n_hidden, self.hyperparameters["dropout_attention"])

        # Define Classifier
        # flattened output of the transformer:
        # 2*(2*(self.max_seq_length+1)+3)*self.embed_dim
        self.classifier_hidden = 64
        self.classifier_in = (2*(max_seq_length)+3)*embed_dim
        self.classifier = Classifier(self.classifier_in, self.classifier_hidden, self.hyperparameters["dropout_linear"])
    

    def forward(self, epitope, trb_cdr3, v_beta, j_beta, mhc):
        '''
        print(f"epitope.shape: {epitope.shape}")
        print(f"trb_cdr3.shape: {trb_cdr3.shape}")
        
        print(f"len(v_beta): {len(v_beta)}")
        print(f"len(j_beta): {len(j_beta)}")
        print(f"len(mhc): {len(mhc)}")
        print(f"type(v_beta): {type(v_beta)}")
        print(f"type(j_beta): {type(j_beta)}")
        print(f"type(mhc): {type(mhc)}")
        '''

        trb_epitope = torch.cat([trb_cdr3, epitope], dim=1)
        # print(f"trb_epitope.shape: {trb_epitope.shape}")

        trb_v_embed = self.trbV_embed(torch.tensor(v_beta).to(self.device_)).unsqueeze(0).permute(1, 0, 2)
        trb_j_embed = self.trbJ_embed(torch.tensor(j_beta).to(self.device_)).unsqueeze(0).permute(1, 0, 2)
        mhc_embed = self.mhc_embed(torch.tensor(mhc).to(self.device_)).unsqueeze(0).permute(1, 0, 2)
        
        '''
        print(f"trb_v_embed: {trb_v_embed.shape}")
        print(f"trb_j_embed: {trb_j_embed.shape}")
        print(f"mhc_embed: {mhc_embed.shape}")
        '''

        trb_epitope_vj_mhc = torch.cat([trb_epitope, trb_v_embed, trb_j_embed, mhc_embed], dim=1)
        # print(f"trb_epitope_vj_mhc.shape: {trb_epitope_vj_mhc.shape}")

        trb_epitope_vj_mhc_attention = self.multihead_attn_global(trb_epitope_vj_mhc) 
        # print(f"trb_epitope_vj_mhc_attention.shape: {trb_epitope_vj_mhc_attention.shape}")       

        trb_epitope_vj_mhc_flatten = trb_epitope_vj_mhc_attention.view(trb_epitope_vj_mhc_attention.size(0), -1)
        # print(f"trb_epitope_vj_mhc_flatten.shape: {trb_epitope_vj_mhc_flatten.shape}")
        
        logits = self.classifier(trb_epitope_vj_mhc_flatten)
        # print(f"logits: {logits}")
        return logits
    

    def training_step(self, batch, batch_idx):
        epitope_embedding = batch["epitope_embedding"]
        trb_cdr3_embedding = batch["trb_cdr3_embedding"]
        v_beta = batch["v_beta"]
        j_beta = batch["j_beta"]
        mhc = batch["mhc"]
        label = batch["label"]
        
        output = self(epitope_embedding, trb_cdr3_embedding, v_beta, j_beta, mhc).squeeze()

        # print(f"this is y in training_step: {y}\n\n\n")
        loss = F.binary_cross_entropy_with_logits(output, label)
        self.log("train_loss", loss, on_step=True, prog_bar=True, batch_size=len(batch))
        return loss


    def test_step(self, batch, batch_idx):
        epitope_embedding = batch["epitope_embedding"]
        trb_cdr3_embedding = batch["trb_cdr3_embedding"]
        v_beta = batch["v_beta"]
        j_beta = batch["j_beta"]
        mhc = batch["mhc"]
        task = batch["task"]
        # print(f"test_step: task: {task}")
        label = batch["label"]

        source = batch.get("source", ["Unknown"] * len(label)) 
        self.sources.extend(source)
        
        output = self(epitope_embedding, trb_cdr3_embedding, v_beta, j_beta, mhc).squeeze(1)
        prediction = torch.sigmoid(output)
        self.test_predictions.append(prediction)
        self.test_labels.append(label)
        self.test_tasks.append(task[0])

        test_loss = F.binary_cross_entropy_with_logits(output, label)
        self.log("test_loss", test_loss, batch_size=len(batch))
        
        return test_loss, prediction, label
    

    def on_test_epoch_end(self):
        test_predictions = torch.stack(self.test_predictions)
        test_labels = torch.stack(self.test_labels)
        # print(f"on_test_epoch_end, test_labels: {test_labels}")
        test_tasks = self.test_tasks
    
        tpp_1 = []
        tpp_2 = []
        tpp_3 = []

        for i, task in enumerate(test_tasks):
            if task == "TPP1": 
                # print(f"task 1: {task}")
                tpp_1.append((test_predictions[i], test_labels[i]))
            if task == "TPP2": 
                # print(f"task 2: {task}")
                tpp_2.append((test_predictions[i], test_labels[i]))
            if task == "TPP3": 
                # print(f"task 3: {task}")
                tpp_3.append((test_predictions[i], test_labels[i]))
 
        self.log("ROCAUC_Test_global", self.auroc(test_predictions, test_labels), prog_bar=True)
        self.log("AP_Test_global", self.avg_precision(test_predictions, test_labels.to(torch.long)), prog_bar=True)

        self.log("ROCAUC_Test_TPP1", self.auroc(torch.tensor([item[0] for item in tpp_1]), torch.tensor([item[1] for item in tpp_1]).to(torch.long)), prog_bar=True)
        self.log("AP_Test_TPP1", self.avg_precision(torch.tensor([item[0] for item in tpp_1]), torch.tensor([item[1] for item in tpp_1]).to(torch.long)), prog_bar=True) 
        
        self.log("ROCAUC_Test_TPP2", self.auroc(torch.tensor([item[0] for item in tpp_2]), torch.tensor([item[1] for item in tpp_2]).to(torch.long)), prog_bar=True)
        self.log("AP_Test_TPP2", self.avg_precision(torch.tensor([item[0] for item in tpp_2]), torch.tensor([item[1] for item in tpp_2]).to(torch.long)), prog_bar=True)          
        
        self.log("ROCAUC_Test_TPP3", self.auroc(torch.tensor([item[0] for item in tpp_3]), torch.tensor([item[1] for item in tpp_3]).to(torch.long)), prog_bar=True)
        self.log("AP_Test_TPP3", self.avg_precision(torch.tensor([item[0] for item in tpp_3]), torch.tensor([item[1] for item in tpp_3]).to(torch.long)), prog_bar=True)


        # Quelleninformationen verarbeiten
        if hasattr(self, "sources") and len(self.sources) == len(test_labels):
            test_sources = np.array(self.sources)
        else:
            test_sources = np.array(["Unknown"] * len(test_labels))
            print("Warnung: 'sources' wurde nicht korrekt initialisiert. Standardwerte werden verwendet.")


        for source in ["10X", "BA"]:
            source_mask = (np.array(self.sources) == source)  # Filter für diese Quelle
            if np.sum(source_mask) == 0:
                continue  # Überspringen, wenn keine Daten vorhanden sind
        
            source_labels = test_labels[source_mask]
            source_predictions = test_predictions[source_mask]
        
            # Umwandlung von Vorhersagen in binäre Werte (Threshold bei 0.5)
            binary_predictions = (source_predictions >= 0.5).astype(int)
        
            # Berechnung von Precision und Recall
            precision = precision_score(source_labels, binary_predictions, zero_division=0)
            recall = recall_score(source_labels, binary_predictions, zero_division=0)
        
            # Berechnung der Confusion Matrix
            conf_matrix_source = confusion_matrix(source_labels, binary_predictions, labels=[0, 1])
        
            # Extrahieren der True Negatives (TN) und false Negatives (FN)
            tn_source = conf_matrix_source[0, 0] if conf_matrix_source.shape == (2, 2) else 0  # Absicherung bei einer einzelnen Klasse
            fn_source = conf_matrix_source[1, 0] if conf_matrix_source.shape == (2, 2) else 0
            fp_source = conf_matrix_source[0, 1] if conf_matrix_source.shape == (2, 2) else 0

            # Berechnung der Negative Prediction Rate (NPR)
            npr_source = tn_source / (tn_source + fp_source) if (tn_source + fp_source) > 0 else 0
            
            # Logging der NPR in W&B
            wandb.log({
                f"Precision ({source})": precision,
                f"Recall ({source})": recall,
                f"test_true_negatives_{source}": tn_source,
                f"test_false_negatives_{source}": fn_source,
                f"test_negative_prediction_rate_{source}": npr_source
            })

        self.test_predictions.clear()
        self.test_labels.clear()
        self.test_tasks.clear()
        self.sources.clear()


    def validation_step(self, batch, batch_idx):
        epitope_embedding = batch["epitope_embedding"]
        trb_cdr3_embedding = batch["trb_cdr3_embedding"]
        v_beta = batch["v_beta"]
        j_beta = batch["j_beta"]
        mhc = batch["mhc"]
        label = batch["label"]

        source = batch.get("source", ["Unknown"] * len(label)) 
        self.sources.extend(source)
            
        output = self(epitope_embedding, trb_cdr3_embedding, v_beta, j_beta, mhc).squeeze(1)

        val_loss = F.binary_cross_entropy_with_logits(output, label)
        self.log("val_loss", val_loss, batch_size=len(batch))
        
        prediction = torch.sigmoid(output)
        predicted_classes = (prediction > 0.5).int()
        label_classes = label.int()

        # Speichere Vorhersagen und Labels
        if not hasattr(self, "val_predictions"):
            self.val_predictions = []
            self.val_labels = []
        self.val_predictions.append(predicted_classes.cpu())
        self.val_labels.append(label_classes.cpu())
        
        self.log("ROCAUC_Val", self.auroc(prediction, label), on_epoch=True, prog_bar=True, batch_size=len(batch))
        self.log("AP_Val", self.avg_precision(prediction, label.to(torch.long)), on_epoch=True, prog_bar=True, batch_size=len(batch))

        return val_loss
    
    
    def configure_optimizers(self):
        optimizer = self.hyperparameters["optimizer"]
        learning_rate = self.hyperparameters["learning_rate"]
        weight_decay = self.hyperparameters["weight_decay"]
        betas = (0.9, 0.98)
        
        if optimizer == "sgd": 
            optimizer = torch.optim.SGD(self.parameters(),
                        lr=learning_rate, momentum=0.9)        
        if optimizer == "adam": 
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)
        else: 
            print("OPTIMIZER NOT FOUND")
        
        return optimizer


    def on_validation_epoch_end(self):
        # Sammle Vorhersagen und Labels
        all_predictions = torch.cat(self.val_predictions).numpy()
        all_labels = torch.cat(self.val_labels).numpy()

        # Berechnung von Precision und Confusion Matrix
        precision = precision_score(all_labels, all_predictions, zero_division=0)
        conf_matrix = confusion_matrix(all_labels, all_predictions)
    
        # Logging in W&B
        wandb.log({
            "val_precision": precision,
            "val_confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=all_labels,
                preds=all_predictions,
                class_names=["Not Binding", "Binding"]
            ),
        })
    
        # Log Precision separat
        self.log("val_precision", precision, on_epoch=True, prog_bar=True)

    
        # Prüfen, ob Quelleninformationen vorhanden sind
        if hasattr(self, "sources") and len(self.sources) == len(all_labels):
            val_sources = np.array(self.sources)
        else:
            val_sources = np.array(["Unknown"] * len(all_labels))
            print("Warnung: 'sources' wurde nicht korrekt initialisiert. Standardwerte werden verwendet.")
        
        # Filter predictions and labels for negatives and by source
        #is_negative = (all_labels == 0)

        # Berechnung der Metriken für jede Quelle
        for source in ["10X", "BA"]:
            print("sources check: ", self.sources[:5])
            source_mask = (np.array(self.sources) == source) #& is_negative 
            print(f"Source: {source}, Source Mask: {np.sum(source_mask)}")

            if np.sum(source_mask) == 0:
                continue  # Überspringen, wenn keine Daten vorhanden sind
    
            source_labels = all_labels[source_mask]
            source_predictions = all_predictions[source_mask]
            print(f"Source: {source}, Predictions Range: {source_predictions.min()} - {source_predictions.max()}")
            print(f"Source: {source}, Labels: {np.unique(source_labels, return_counts=True)}")
    
            # Berechnung von Precision und Recall
            precision = precision_score(source_labels, (source_predictions > 0.5), zero_division=0)
            recall = recall_score(source_labels, (source_predictions > 0.5), zero_division=0)

            tp = np.sum((source_predictions > 0.5) & (source_labels == 1))
            fp = np.sum((source_predictions > 0.5) & (source_labels == 0))
            fn = np.sum((source_predictions <= 0.5) & (source_labels == 1))
            precision_manual = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall_manual = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            print(f"Source: {source}, TP: {tp}, FP: {fp}, FN: {fn}, Precision: {precision_manual}, Recall: {recall_manual}")
            
    
            # Loggen der Werte in W&B
            wandb.log({
                f"Precision ({source})": precision,
                f"Recall ({source})": recall,
            }, commit=False)

            # Berechnung der Confusion Matrix
            conf_matrix_source = confusion_matrix(source_labels, source_predictions, labels=[0, 1])
            tn_source = conf_matrix_source[0, 0] if conf_matrix_source.shape == (2, 2) else 0  # Absicherung bei einer einzelnen Klasse
            fn_source = conf_matrix_source[1, 0] if conf_matrix_source.shape == (2, 2) else 0
            fp_source = conf_matrix_source[0, 1] if conf_matrix_source.shape == (2, 2) else 0
            
            # Berechnung der Negative Prediction Rate (NPR)
            npr_source = tn_source / (tn_source + fp_source) if (tn_source + fp_source) > 0 else 0
            
            # Logging der NPR in W&B
            wandb.log({
                f"val_true_negatives_{source}": tn_source,
                f"val_false_positives_{source}": fp_source,
                f"val_precision_{source}": precision_score(source_labels, source_predictions, zero_division=0),
                f"val_negative_prediction_rate_{source}": npr_source
            })


        positive_preds = all_predictions[all_labels == 1]
        negative_preds = all_predictions[all_labels == 0]
    
        # Histogramm plotten
        plt.figure(figsize=(10, 6))
        plt.hist(negative_preds, bins=50, alpha=0.7, label="Negative Predictions")
        plt.hist(positive_preds, bins=50, alpha=0.7, label="Positive Predictions")
        plt.xlabel("Sigmoid Output")
        plt.ylabel("Frequency")
        plt.title("Prediction Distribution (Validation)")
        plt.legend()
        plt.show()
    
        # Logging in W&B (optional)
        wandb.log({"validation_prediction_histogram": plt})
    
        # Cleanup für die nächste Epoche
        self.val_predictions.clear()
        self.val_labels.clear()
        self.sources = []