# Corpus (lu dans traduction_.csv), tokenisation et vocabulaire
import csv
import os
import re

import torch

PAD, SOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<sos>", "<eos>", "<unk>"]

# le fichier csv est à la racine du projet (à côté de main.py)
CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "traduction_.csv")
# noms de colonnes acceptés pour la ligne d'en-tête (elle est ignorée)
ENTETES = {"en", "english", "anglais", "fr", "french", "francais", "français"}


def build_corpus(path=CSV_PATH):
    # une paire par ligne : phrase anglaise ; phrase française (séparateur , ou ; ou tabulation)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}\n"
                                "Il faut un csv avec deux colonnes : anglais, français.")
    with open(path, newline="", encoding="utf-8-sig") as f:
        extrait = f.read(2048)
        f.seek(0)
        try:
            dialecte = csv.Sniffer().sniff(extrait, delimiters=",;\t")
        except csv.Error:
            dialecte = csv.excel
        lignes = list(csv.reader(f, dialecte))

    pairs = []
    for ligne in lignes:
        ligne = [c.strip() for c in ligne]
        if len(ligne) < 2 or not ligne[0] or not ligne[1]:
            continue
        if not pairs and ligne[0].lower() in ENTETES:
            continue  # en-tête
        pairs.append((ligne[0], ligne[1]))
    if not pairs:
        raise ValueError(f"Aucune paire de phrases trouvée dans {path}")
    return pairs


# un mot, un mot avec apostrophe (j'), ou un signe de ponctuation
TOKEN_RE = re.compile(r"\w+'|\w+|[^\w\s]", re.UNICODE)


def tokenize(s):
    return TOKEN_RE.findall(s.lower())


def detokenize(tokens):
    s = " ".join(t for t in tokens if t not in SPECIALS)
    s = re.sub(r"(\w') ", r"\1", s)
    s = s.replace(" .", ".").replace(" ,", ",").replace(" ?", "?").replace(" !", "!")
    return s


class Vocab:
    def __init__(self, sentences):
        self.itos = list(SPECIALS)
        for s in sentences:
            for t in tokenize(s):
                if t not in self.itos:
                    self.itos.append(t)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def ids(self, tokens):
        return [self.stoi.get(t, UNK) for t in tokens]


def make_tensors(pairs, sv, tv):
    # source : phrase + <eos> / cible : <sos> + phrase + <eos>, complétées avec <pad>
    src = [sv.ids(tokenize(e)) + [EOS] for e, _ in pairs]
    tgt = [[SOS] + tv.ids(tokenize(f)) + [EOS] for _, f in pairs]
    ls, lt = max(map(len, src)), max(map(len, tgt))
    S = torch.full((len(src), ls), PAD, dtype=torch.long)
    T = torch.full((len(tgt), lt), PAD, dtype=torch.long)
    for i, (a, b) in enumerate(zip(src, tgt)):
        S[i, :len(a)] = torch.tensor(a)
        T[i, :len(b)] = torch.tensor(b)
    return S, T
