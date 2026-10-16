# Fenêtre principale + onglet 1 (configuration et entraînement)
import queue
import random
import threading
import time

import torch
import torch.nn.functional as F

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from . import textes as tx
from .data import PAD, EOS, build_corpus, Vocab, make_tensors, tokenize
from .model import Transformer, make_src_mask, make_tgt_mask, greedy_translate
from .plots import (plt, BG, BG2, BG3, FG, SEL, make_canvas, finish, dark_text)
from .tabs import VectorTab, EncoderTab, DecoderTab, DiagTab

# champs de configuration : (clé, valeur par défaut, type)
CFG_FIELDS = [
    ("d_model", "64", int),
    ("pe", "sinus", str),
    ("n_enc", "2", int),
    ("n_dec", "2", int),
    ("n_heads", "4", int),
    ("d_ff", "128", int),
    ("dropout", "0.1", float),
    ("lr", "0.001", float),
    ("epochs", "30", int),
    ("batch", "32", int),
]
PE_CODES = {label: code for code, label in tx.PE_LABELS.items()}


def fmt_n(n):
    return f"{n:,}".replace(",", " ")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(tx.COMMUN["titre"])
        self.geometry("1500x930")
        self.configure(bg=BG)
        self._style()

        self.pairs = build_corpus()
        self.sv = Vocab([e for e, _ in self.pairs])
        self.tv = Vocab([f for _, f in self.pairs])
        self.src_t, self.tgt_t = make_tensors(self.pairs, self.sv, self.tv)

        self.sentence_var = tk.StringVar(value="she eats an apple .")
        self.fr_var = tk.StringVar(value="")
        self.model, self.cfg = None, None
        self.losses, self.accs, self.runs = [], [], []
        self.q = queue.Queue()  # l'entraînement tourne dans un thread et envoie ses résultats ici
        self.stop_event = threading.Event()
        self.training = False

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)
        tabs = []
        for name in tx.COMMUN["onglets"]:
            fr = ttk.Frame(nb)
            nb.add(fr, text=name)
            tabs.append(fr)
        self._build_config_tab(tabs[0])
        self.tab_en = VectorTab(self, tabs[1], "src")
        self.tab_enc = EncoderTab(self, tabs[2])
        self.tab_dec = DecoderTab(self, tabs[3])
        self.tab_fr = VectorTab(self, tabs[4], "tgt")
        self.tab_diag = DiagTab(self, tabs[5])

        self.status = ttk.Label(self, text="", anchor="w")
        self.status.pack(fill="x", side="bottom", padx=6, pady=2)

        self.reset_model(silent=True)
        self.after(150, self._poll)

    # ------------------------------------------------------------------ style
    def _style(self):
        st = ttk.Style(self)
        st.theme_use("clam")
        st.configure(".", background=BG, foreground=FG, fieldbackground=BG2, bordercolor=BG3,
                     lightcolor=BG3, darkcolor=BG3, troughcolor=BG2, insertcolor=FG)
        st.configure("TFrame", background=BG)
        st.configure("TLabel", background=BG, foreground=FG)
        st.configure("TLabelframe", background=BG, foreground=FG)
        st.configure("TLabelframe.Label", background=BG, foreground=FG)
        st.configure("TButton", background=BG3, foreground=FG, padding=4)
        st.map("TButton", background=[("active", "#4a4a4a"), ("disabled", BG2)],
               foreground=[("disabled", "#777777")])
        st.configure("TEntry", fieldbackground=BG2, foreground=FG)
        st.configure("TCombobox", fieldbackground=BG2, background=BG3, foreground=FG, arrowcolor=FG)
        st.map("TCombobox", fieldbackground=[("readonly", BG2)], foreground=[("readonly", FG)],
               selectbackground=[("readonly", BG2)], selectforeground=[("readonly", FG)])
        st.configure("TNotebook", background=BG, borderwidth=0)
        st.configure("TNotebook.Tab", background=BG2, foreground=FG, padding=(10, 4))
        st.map("TNotebook.Tab", background=[("selected", BG3)])
        st.configure("Treeview", background=BG2, fieldbackground=BG2, foreground=FG, rowheight=20)
        st.configure("Treeview.Heading", background=BG3, foreground=FG)
        st.map("Treeview", background=[("selected", SEL)])
        st.configure("Vertical.TScrollbar", background=BG3, arrowcolor=FG)
        self.option_add("*TCombobox*Listbox.background", BG2)
        self.option_add("*TCombobox*Listbox.foreground", FG)
        self.option_add("*TCombobox*Listbox.selectBackground", SEL)

    # ----------------------------------------------------------- onglet 1
    def _build_config_tab(self, parent):
        t = tx.APP
        left = ttk.Frame(parent, width=380)
        left.pack(side="left", fill="y", padx=8, pady=8)
        left.pack_propagate(False)
        right = ttk.Frame(parent)
        right.pack(side="left", fill="both", expand=True)

        box = ttk.LabelFrame(left, text=t["hyperparams"])
        box.pack(fill="x")
        self.cfg_vars = {}
        for r, (key, default, _) in enumerate(CFG_FIELDS):
            ttk.Label(box, text=t["cfg_labels"][key]).grid(row=r, column=0, sticky="w", padx=4, pady=3)
            if key == "pe":
                v = tk.StringVar(value=tx.PE_LABELS[default])
                w = ttk.Combobox(box, textvariable=v, state="readonly", width=12,
                                 values=list(tx.PE_LABELS.values()))
            else:
                v = tk.StringVar(value=default)
                w = ttk.Entry(box, textvariable=v, width=14)
            w.grid(row=r, column=1, padx=4, pady=3)
            self.cfg_vars[key] = v

        btns = ttk.Frame(left)
        btns.pack(fill="x", pady=8)
        self.btn_train = ttk.Button(btns, text=t["btn_train"], command=self.train)
        self.btn_stop = ttk.Button(btns, text=t["btn_stop"], command=self.stop_event.set, state="disabled")
        self.btn_load = ttk.Button(btns, text=t["btn_load"], command=self.load)
        self.btn_save = ttk.Button(btns, text=t["btn_save"], command=self.save)
        self.btn_reset = ttk.Button(btns, text=t["btn_reset"], command=self.reset_model)
        for i, b in enumerate((self.btn_train, self.btn_stop, self.btn_load, self.btn_save, self.btn_reset)):
            b.grid(row=i // 2, column=i % 2, sticky="ew", padx=2, pady=2)
        btns.columnconfigure(0, weight=1)
        btns.columnconfigure(1, weight=1)

        ttk.Label(left, text=t["journal"]).pack(anchor="w")
        self.log_txt = dark_text(left, height=20)
        self.log_txt.pack(fill="both", expand=True)

        self.fig1, self.canvas1 = make_canvas(right)
        self.log(t["corpus"].format(n=len(self.pairs), ven=len(self.sv), vfr=len(self.tv)))
        self.log(t["exemples"])
        for e, f in random.sample(self.pairs, 6):
            self.log(t["exemple"].format(en=e, fr=f))

    def log(self, s):
        self.log_txt.insert("end", s + "\n")
        self.log_txt.see("end")

    def get_cfg(self):
        t = tx.APP
        cfg = {}
        for key, _, typ in CFG_FIELDS:
            label = t["cfg_labels"][key]
            raw = self.cfg_vars[key].get()
            try:
                cfg[key] = PE_CODES[raw] if key == "pe" else typ(raw)
            except (ValueError, KeyError):
                raise ValueError(t["err_valeur"].format(label=label))
        for k in ("d_model", "n_enc", "n_dec", "n_heads", "d_ff", "epochs", "batch"):
            if cfg[k] <= 0:
                raise ValueError(t["err_positif"].format(label=t["cfg_labels"][k]))
        if cfg["d_model"] % cfg["n_heads"] != 0:
            raise ValueError(t["err_heads"])
        if cfg["pe"] == "sinus" and cfg["d_model"] % 2:
            raise ValueError(t["err_pair"])
        if not 0 <= cfg["dropout"] < 1:
            raise ValueError(t["err_dropout"])
        return cfg

    def build_model(self, cfg):
        return Transformer(len(self.sv), len(self.tv), cfg["d_model"], cfg["n_heads"], cfg["n_enc"],
                           cfg["n_dec"], cfg["d_ff"], cfg["dropout"], cfg["pe"])

    def n_params(self):
        return sum(p.numel() for p in self.model.parameters())

    def reset_model(self, silent=False):
        if self.training:
            return
        try:
            cfg = self.get_cfg()
        except ValueError as e:
            messagebox.showwarning(tx.APP["titre_config"], str(e))
            return
        self.cfg = cfg
        self.model = self.build_model(cfg)
        self.losses, self.accs = [], []
        self.draw_loss()
        if not silent:
            self.log(tx.APP["reset_log"])
        self.set_status(tx.APP["statut_non_entraine"].format(n=fmt_n(self.n_params())))

    def set_status(self, s):
        self.status.configure(text=s)

    def check_idle(self):
        if self.training:
            messagebox.showinfo(tx.APP["titre_patience"], tx.APP["msg_patience"])
            return False
        return True

    # ------------------------------------------------------------ entraînement
    def train(self):
        if self.training:
            return
        t = tx.APP
        try:
            cfg = self.get_cfg()
        except ValueError as e:
            messagebox.showwarning(t["titre_config"], str(e))
            return
        if cfg != self.cfg or self.model is None:
            self.cfg = cfg
            self.model = self.build_model(cfg)
            self.losses, self.accs = [], []
            self.log(t["nouvelle_config"])
        self.training = True
        self.stop_event.clear()
        for b in (self.btn_train, self.btn_load, self.btn_reset, self.btn_save):
            b.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.log(t["log_entrainement"].format(cfg=cfg))
        self.log(t["log_params"].format(n=fmt_n(self.n_params())))
        threading.Thread(target=self._train_worker, args=(self.model, cfg), daemon=True).start()

    def _train_worker(self, model, cfg):
        t0 = time.time()
        try:
            opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.98), eps=1e-9)
            n, V = len(self.src_t), len(self.tv)
            for ep in range(cfg["epochs"]):
                if self.stop_event.is_set():
                    break
                model.train()
                perm = torch.randperm(n)
                tot, correct, total = 0.0, 0, 0
                for i in range(0, n, cfg["batch"]):
                    idx = perm[i:i + cfg["batch"]]
                    s, t = self.src_t[idx], self.tgt_t[idx]
                    logits = model(s, t[:, :-1])  # le decoder reçoit la cible décalée d'un cran
                    y = t[:, 1:]
                    loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), ignore_index=PAD)
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    tot += loss.item() * len(idx)
                    m = y != PAD
                    correct += ((logits.argmax(-1) == y) & m).sum().item()
                    total += m.sum().item()
                self.q.put(("epoch", ep + 1, tot / n, correct / total))
            # à la fin : traduction complète de 100 phrases du corpus pour l'exact match
            sample = random.sample(self.pairs, min(100, len(self.pairs)))
            ok = 0
            for e, f in sample:
                ys, _ = greedy_translate(model, self.sv, self.tv, tokenize(e))
                ok += [self.tv.itos[i] for i in ys[1:-1]] == tokenize(f)
            self.q.put(("done", ok / len(sample), time.time() - t0))
        except Exception as ex:  # noqa
            self.q.put(("error", str(ex)))

    def _poll(self):
        t = tx.APP
        try:
            while True:
                msg = self.q.get_nowait()
                if msg[0] == "epoch":
                    _, ep, loss, acc = msg
                    self.losses.append(loss)
                    self.accs.append(acc)
                    self.log(t["log_epoch"].format(ep=ep, loss=loss, acc=acc))
                    self.set_status(t["statut_epoch"].format(ep=ep, total=self.cfg["epochs"], loss=loss))
                    self.draw_loss()
                elif msg[0] == "done":
                    _, exact, dt = msg
                    self._end_training()
                    self.runs.append({"id": len(self.runs) + 1, "cfg": dict(self.cfg), "losses": list(self.losses),
                                      "acc": self.accs[-1] if self.accs else 0.0, "exact": exact, "time": dt})
                    self.log(t["log_fini"].format(dt=dt, exact=exact))
                    last = self.losses[-1] if self.losses else float("nan")
                    self.set_status(t["statut_fini"].format(loss=last, exact=exact))
                    self.tab_diag.refresh()
                elif msg[0] == "error":
                    self._end_training()
                    messagebox.showerror(t["titre_erreur"], msg[1])
        except queue.Empty:
            pass
        self.after(150, self._poll)

    def _end_training(self):
        self.training = False
        for b in (self.btn_train, self.btn_load, self.btn_reset, self.btn_save):
            b.configure(state="normal")
        self.btn_stop.configure(state="disabled")

    def draw_loss(self):
        t = tx.APP
        f = self.fig1
        f.clear()
        ax = f.add_subplot(1, 1, 1)
        if self.losses:
            ax.plot(range(1, len(self.losses) + 1), self.losses, marker=".", label=t["courbe_loss"])
            ax2 = ax.twinx()
            ax2.plot(range(1, len(self.accs) + 1), self.accs, color="#ffa64f", label=t["courbe_acc"])
            ax2.set_ylim(0, 1)
            ax.legend(loc="upper center")
            ax2.legend(loc="center right")
        else:
            ax.text(0.5, 0.5, t["pas_entraine"], ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel(tx.COMMUN["epoch"])
        ax.set_title(t["courbe_titre"])
        finish(f, self.canvas1)

    # ------------------------------------------------------- sauvegarde / chargement
    def save(self):
        path = filedialog.asksaveasfilename(defaultextension=".pt", filetypes=[("PyTorch", "*.pt")])
        if not path:
            return
        torch.save({"cfg": self.cfg, "state": self.model.state_dict(), "losses": self.losses,
                    "accs": self.accs, "src_itos": self.sv.itos, "tgt_itos": self.tv.itos}, path)
        self.log(tx.APP["log_sauvegarde"].format(path=path))

    def load(self):
        t = tx.APP
        path = filedialog.askopenfilename(filetypes=[("PyTorch", "*.pt"), ("*", "*.*")])
        if not path:
            return
        try:
            try:
                ck = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                ck = torch.load(path, map_location="cpu")
            if ck["src_itos"] != self.sv.itos or ck["tgt_itos"] != self.tv.itos:
                raise ValueError(t["err_vocab"])
            for k, v in ck["cfg"].items():
                self.cfg_vars[k].set(tx.PE_LABELS[v] if k == "pe" else str(v))
            self.cfg = ck["cfg"]
            self.model = self.build_model(self.cfg)
            self.model.load_state_dict(ck["state"])
            self.losses, self.accs = list(ck["losses"]), list(ck["accs"])
            self.draw_loss()
            self.log(t["log_charge"].format(path=path))
            self.set_status(t["statut_charge"].format(n=fmt_n(self.n_params())))
        except Exception as e:
            messagebox.showerror(t["titre_chargement"], str(e))

    # ------------------------------------------------ passages du modèle pour les graphiques
    @torch.no_grad()
    def encode_sentence(self, sentence):
        words = tokenize(sentence)
        if not words:
            raise ValueError(tx.COMMUN["phrase_en_vide"])
        ids = self.sv.ids(words) + [EOS]
        src = torch.tensor([ids])
        m = self.model
        m.eval()
        mask = make_src_mask(src)
        mem = m.encode(src, mask)
        tr = m.trace
        return {
            "tokens": [w if w in self.sv.stoi else "<unk>" for w in words] + ["<eos>"],
            "raw": words + ["<eos>"], "ids": ids, "mask": mask, "memory": mem,
            "emb": tr["src_emb"][0].numpy(), "pe": tr["src_pe"][0].numpy(),
            "layers": [o[0].numpy() for o in tr["enc_outputs"]],
            "attn": [{k: v[0].numpy() for k, v in L.attn.last.items()} for L in m.enc_layers],
        }

    @torch.no_grad()
    def decode_prefix(self, enc, ids):
        m = self.model
        m.eval()
        tgt = torch.tensor([ids])
        logits = m.decode(tgt, enc["memory"], enc["mask"], make_tgt_mask(tgt))
        tr = m.trace
        return {
            "probs": torch.softmax(logits[0], -1).numpy(),
            "emb": tr["tgt_emb"][0].numpy(), "pe": tr["tgt_pe"][0].numpy(),
            "layers": [o[0].numpy() for o in tr["dec_outputs"]],
            "self_attn": [{k: v[0].numpy() for k, v in L.self_attn.last.items()} for L in m.dec_layers],
            "cross_attn": [{k: v[0].numpy() for k, v in L.cross_attn.last.items()} for L in m.dec_layers],
        }
