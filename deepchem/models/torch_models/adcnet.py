from typing import List
import torch
import torch.nn as nn
from transformers import EsmTokenizer, EsmModel, AutoTokenizer, AutoModel


class ADCNet(nn.Module):
    """
    ADCNet is a neural network model designed for antibody-drug conjugate (ADC) analysis,
    leveraging transformer-based embeddings to capture molecular and protein interactions. 
    It takes as input protein sequences (heavy chain, light chain, and antigen) and chemical
    structures (SMILES strings for payload and linker molecules), along with the drug-to-antibody
    ratio (DAR). ADCNet uses pre-trained ESM-2 to generate embeddings for protein sequences and
    ChemBERTa for SMILES strings, concatenating these with the DAR to form a comprehensive feature representation. 
    These features are processed through a multilayer perceptron (MLP) to predict properties of the ADC, 
    therapeutic efficacy. The model iteratively refines the combined 
    representation of protein and chemical components to produce a final output.
    
    Examples: 
    
    """
    def __init__(self,
                 seq_model_name: str = 'facebook/esm2_t6_8M_UR50D',
                 chemberta_task: str = 'regression',
                 chemberta_tokenizer: str = 'seyonec/ChemBERTa-zinc-base-v1',
                 hidden_dim: int = 128,
                 output_dim: int = 1):
        """
        Args:
            seq_model_name: HuggingFace name for ESM-2 (e.g. 'facebook/esm2_t6_8M_UR50D')
            chemberta_task: one of ['mlm','mtr','regression','classification']
            chemberta_tokenizer: HF path for the ChemBERTa tokenizer
            hidden_dim: width of the MLP hidden layer
            output_dim: final output dimension (e.g. number of regression targets)
        """
        super(ADCNet, self).__init__()
        
        # ---- ESM-2 embedder ----
        self.esm_tokenizer = EsmTokenizer.from_pretrained(seq_model_name)
        self.esm_model     = EsmModel.from_pretrained(seq_model_name)
        
        self.esm_model.eval()
        for p in self.esm_model.parameters():
            p.requires_grad = False


        # ---- Chemberta embedder ----
        self.chemberta_tokenizer = AutoTokenizer.from_pretrained(chemberta_tokenizer)
        self.chemberta_model     = AutoModel.from_pretrained(chemberta_tokenizer)
        
        self.chemberta_model.eval()
        for p in self.chemberta_model.parameters():
            p.requires_grad = False


        # ---- downstream MLP ----
        # ESM2 hidden size + ChemBERTa hidden size =  512 (for esm2_t6) + 768 = 1280
        seq_emb_dim = self.esm_model.config.hidden_size
        chem_emb_dim = self.chemberta_model.config.hidden_size
        total_dim = seq_emb_dim * 3 + chem_emb_dim * 2 + 1

        self.fc1 = nn.Linear(total_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

    def _embed_sequence(self, seq_batch: List[str]) -> torch.Tensor:
        """
        Tokenize protein sequences and return the CLS (first token) embedding.
        seq_batch: list of strings, length B
        returns: (B, hidden_size)
        """
        tokens = self.esm_tokenizer(seq_batch,
                                    return_tensors="pt",
                                    padding=True,
                                    truncation=True)
        tokens = {k: v.to(self.esm_model.device) for k, v in tokens.items()}
        with torch.no_grad():
            out = self.esm_model(**tokens).last_hidden_state
        return out[:, 0, :]

    def _embed_smiles(self, smiles_batch: List[str]) -> torch.Tensor:
        """
        Use Chemberta to embed SMILES.
        smiles_batch: list of strings, length B
        returns: (B, hidden_size)
        """
        tokens = self.chemberta_tokenizer(smiles_batch,
                                          return_tensors="pt",
                                          padding=True,
                                          truncation=True)
        tokens = {k: v.to(self.chemberta_model.device) for k, v in tokens.items()}
        with torch.no_grad():
            out = self.chemberta_model(**tokens).last_hidden_state
        return out[:, 0, :]

    def forward(self,
                sequence_heavy: List[str],
                sequence_light: List[str],
                sequence_antigen: List[str],                
                smiles_payload: List[str],
                smiles_linker: List[str],
                dar: torch.Tensor) -> torch.Tensor:
        """
        Args:
          seqs: list of protein sequences (antibody+antigen concatenated, or pass separately)
          smiles_payload: list of payload SMILES
          smiles_linker: list of linker SMILES
        """
        # get embeddings
        heavy_emb    = self._embed_sequence(sequence_heavy)             # (B, seq_emb_dim)
        light_emb    = self._embed_sequence(sequence_light)             # (B, seq_emb_dim)
        antigen_emb    = self._embed_sequence(sequence_antigen)         # (B, seq_emb_dim)
        payload_emb    = self._embed_smiles(smiles_payload)             # (B, chem_emb_dim)
        linker_emb   = self._embed_smiles(smiles_linker)                # (B, chem_emb_dim)

        # concat
        x = torch.cat([heavy_emb, light_emb, antigen_emb, payload_emb, linker_emb, dar])   # (B, total_dim)

        # downstream MLP
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        out = self.fc3(x)
        return out