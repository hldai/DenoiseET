CUDA_VISIBLE_DEVICES=3 python main.py labeler -enhanced_mention -data_setup joint -add_crowd -multitask -mention_lstm -add_headword_emb -model_type labeler -remove_el -remove_open -mode train_labeler