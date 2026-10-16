# Tous les textes de l'interface sont ici.
# Les {accolades} sont remplies par le code, il ne faut pas les enlever.

# ---------------------------------------------------------------- communs
COMMUN = {
    "titre": "Transformer EN → FR fait main (PyTorch) – visualisation",
    "onglets": [
        "1. Configuration et entraînement",
        "2. Texte anglais et vectorisation",
        "3. Encoder : trajectoire",
        "4. Decoder : attention et génération",
        "5. Vectorisation française",
        "6. Traduction et diagnostic",
    ],
    "attention": "Attention",
    "phrase_en_vide": "Phrase anglaise vide.",
    "phrase_fr_vide": "Phrase française vide.",
    "phrase_en": "Phrase anglaise :",
    "couche": "Couche {i}",
    "tete": "Tête {i}",
    "epoch": "epoch",
    "embedding": "embedding",
    "emb_pe": "emb + PE",
    "pos_enc": "pos. encoding",
    "acc_tokens": "acc. tokens",
    "loss": "loss",
}

# types d'encodage positionnel : code interne -> texte affiché
PE_LABELS = {"sinus": "Sinusoïdal", "appris": "Appris", "aucun": "Aucun"}

# ---------------------------------------------------------------- onglet 1
APP = {
    "cfg_labels": {
        "d_model": "Dimension des embeddings (d_model)",
        "pe": "Type d'embeddings positionnels",
        "n_enc": "Profondeur de l'encoder (couches)",
        "n_dec": "Profondeur du decoder (couches)",
        "n_heads": "Nombre de têtes",
        "d_ff": "Dimension du feed-forward",
        "dropout": "Dropout",
        "lr": "Learning rate",
        "epochs": "Epochs",
        "batch": "Batch size",
    },
    "hyperparams": "Hyperparamètres",
    "btn_train": "Entraîner",
    "btn_stop": "Arrêter",
    "btn_load": "Charger…",
    "btn_save": "Sauvegarder…",
    "btn_reset": "Réinitialiser",
    "journal": "Journal :",
    "corpus": "Corpus : {n} paires | vocab EN = {ven} | vocab FR = {vfr}",
    "exemples": "Exemples :",
    "exemple": "  {en}  ->  {fr}",
    # erreurs de configuration
    "titre_config": "Configuration",
    "err_valeur": "Valeur invalide pour « {label} ».",
    "err_positif": "« {label} » doit être > 0.",
    "err_heads": "d_model doit être divisible par le nombre de têtes.",
    "err_pair": "d_model doit être pair avec le positional encoding sinusoïdal.",
    "err_dropout": "Le dropout doit être dans [0, 1[.",
    # messages
    "reset_log": "Modèle réinitialisé (poids aléatoires).",
    "statut_non_entraine": "Modèle NON entraîné – {n} paramètres",
    "titre_patience": "Patience",
    "msg_patience": "Entraînement en cours, attends la fin ou clique sur Arrêter.",
    "nouvelle_config": "Nouvelle configuration -> nouveau modèle.",
    "log_entrainement": "Entraînement : {cfg}",
    "log_params": "Paramètres : {n}",
    "log_epoch": "epoch {ep:3d} | loss {loss:.4f} | acc. tokens {acc:.3f}",
    "statut_epoch": "Entraînement… epoch {ep}/{total} – loss {loss:.4f}",
    "log_fini": "Terminé en {dt:.1f}s – exact match (100 phrases) : {exact:.2%}",
    "statut_fini": "Modèle entraîné – loss {loss:.4f} – exact match {exact:.0%}",
    "titre_erreur": "Erreur",
    # courbe
    "courbe_loss": "loss (cross-entropy)",
    "courbe_acc": "précision tokens",
    "pas_entraine": "Pas encore entraîné",
    "courbe_titre": "Courbe d'entraînement",
    # sauvegarde / chargement
    "log_sauvegarde": "Modèle sauvegardé : {path}",
    "titre_chargement": "Chargement",
    "err_vocab": "Le vocabulaire du fichier ne correspond pas au corpus actuel.",
    "log_charge": "Modèle chargé : {path}",
    "statut_charge": "Modèle chargé – {n} paramètres",
}

# ---------------------------------------------------------------- onglets 2 et 5
VEC = {
    "label_en": "Phrase anglaise :",
    "label_fr": "Phrase française :",
    "btn_vectoriser": "Tokeniser et vectoriser",
    "btn_traduction": "Utiliser la traduction du modèle",
    "note_fr": "(le decoder est conditionné par la\nphrase anglaise de l'onglet 2)",
    "label_tokens": "Tokens / IDs (cliquer pour inspecter) :",
    "cols_tokens": ("pos", "token", "id"),
    "label_inspection": "Inspection du token :",
    "unk": "{r} (→<unk>)",
    "sortie_encoder": "sortie encoder",
    "sortie_decoder": "sortie decoder",
    "inspection": (
        "Token     : {tok}\n"
        "ID        : {id} ({itos})\n"
        "Position  : {pos}\n"
        "d_model   : {d}\n\n"
        "||embedding||  = {n_e:.3f}\n"
        "||pos. enc.||  = {n_p:.3f}\n"
        "||emb + PE||    = {n_ep:.3f}\n"
        "||{out}|| = {n_o:.3f}\n\n"
        "Embedding [0:8] :\n{e8}\n\n"
        "PE [0:8] :\n{p8}\n\n"
        "Voisins les plus proches\n(cosinus dans la table d'embeddings) :\n"
    ),
    "plot_emb": "Embeddings (×√d_model)",
    "plot_pe": "Positional encoding",
    "plot_pca": "Projection PCA 3D",
    "plot_token": "Vecteurs du token « {tok} »",
    "plot_dim": "dimension",
}

# ---------------------------------------------------------------- onglet 3
ENC = {
    "label_phrase": "Phrase (onglet 2) :",
    "btn_encoder": "Encoder la phrase",
    "label_couche": "Couche (attention) :",
    "label_tete": "Tête :",
    "label_token": "Token sélectionné :",
    "label_infos": "Infos :",
    "pages": ("Trajectoire des vecteurs", "Q / K / V et scores", "Toutes les têtes"),
    "nom_entree": "entrée",
    "nom_couche": "couche {i}",
    "info_haut": "Token : « {tok} »\n\nNorme par couche :\n",
    "info_ligne_norme": "  {nm:<9} {nv:7.3f}  cos(entrée)={c:+.3f}",
    "info_attention": "\n\nCouche {l}, tête {h}\nd_k = {dk}\nAttention de « {tok} » vers :\n",
    "info_ligne_attn": "  {tok:<10} {w:.3f}",
    "plot_trajectoire": "Trajectoire à travers les couches\n(PCA commune, chiffres = couche)",
    "plot_vecteur": "Vecteur de « {tok} » par couche",
    "plot_norme": "norme",
    "plot_cos": "cos(entrée)",
    "plot_evolution": "Évolution de « {tok} »",
    "plot_q": "Q (requêtes)",
    "plot_k": "K (clés)",
    "plot_v": "V (valeurs)",
    "plot_scores": "Scores QKᵀ/√d_k",
    "plot_poids": "Poids d'attention (softmax)",
    "plot_regarde": "« {tok} » regarde…",
    "plot_titre_page2": "Encoder – couche {l}, tête {h}",
    "plot_tete": "Tête {k}",
    "plot_titre_page3": "Poids d'attention de toutes les têtes – couche {l}",
}

# ---------------------------------------------------------------- onglet 4
DEC = {
    "label_strategie": "Stratégie :",
    "strategies": ["Greedy (argmax)", "Échantillonnage (T=1)"],
    "btn_demarrer": "Démarrer (encoder + <sos>)",
    "btn_etape": "Étape suivante  ▶",
    "btn_tout": "Générer jusqu'à <eos>  ▶▶",
    "label_couche": "Couche du decoder :",
    "label_tete": "Tête :",
    "label_qkv": "Q/K/V affichés :",
    "kinds": ["Masked self-attention", "Cross-attention"],
    "label_generes": "Tokens français générés :",
    "label_top": "Prochain token – top 10 :",
    "fini": "[génération terminée]",
    "distrib": "(distribution après le dernier token)\n\n",
    "distrib_fin": "(séquence terminée)\n\n",
    "ligne_top": "{r:>2}. {tok:<14} {p:6.2f} %",
    "plot_self": "Masked self-attention",
    "plot_cross": "Cross-attention → encoder",
    "plot_proba": "P(prochain token) – top 10",
    "plot_q": "Q (decoder)",
    "plot_k": "K ({qui})",
    "plot_v": "V ({qui})",
    "qui_decoder": "decoder",
    "qui_encoder": "encoder",
    "plot_titre": "Decoder – couche {l}, tête {h} – étape {s}",
}

# ---------------------------------------------------------------- onglet 6
DIAG = {
    "btn_traduire": "Traduire",
    "btn_effacer": "Effacer l'historique des configurations",
    "label_traduction": "Traduction finale :",
    "label_pas_a_pas": "Génération token par token :",
    "cols_pas": ("étape", "token", "proba", "alternatives"),
    "label_comparaison": "Comparaison des configurations entraînées :",
    "cols_runs": ("n°", "d_model", "PE", "enc/dec", "têtes", "d_ff", "dropout", "lr", "epochs",
                  "batch", "loss finale", "acc. tokens", "exact match", "durée"),
    "inconnus": "\n(mots inconnus : {mots})",
    "plot_proba": "Probabilité du token choisi à chaque étape",
    "plot_loss": "Loss d'entraînement (modèle actuel)",
    "plot_cmp_loss": "Comparaison : courbes de loss",
    "plot_cmp_acc": "Comparaison : précision",
    "exact_match": "exact match",
}
