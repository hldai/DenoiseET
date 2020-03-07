import os
import json
import config


def __apply_denoise_results(raw_data_file, filter_pred_file, relabel_pred_file, output_file):
    with open(filter_pred_file, encoding='utf-8') as f:
        filter_pred_dict = json.loads(next(f))
    print(len(filter_pred_dict))

    with open(relabel_pred_file, encoding='utf-8') as f:
        relabel_pred_dict = json.loads(next(f))
    print(len(relabel_pred_dict))

    cnt, nfcnt = 0, 0
    f = open(raw_data_file, encoding='utf-8')
    fout = open(output_file, 'w', encoding='utf-8', newline='\n')
    for i, line in enumerate(f):
        example = json.loads(line)
        annot_id = example['annot_id']
        filter_pred = filter_pred_dict.get(annot_id)
        if filter_pred is None:
            nfcnt += 1
            continue
        else:
            filter_pred = filter_pred['cls_pred']
        if filter_pred == 1:
            continue

        relabel_pred = relabel_pred_dict[annot_id]['pred']
        example['y_str'] = relabel_pred
        fout.write('{}\n'.format(json.dumps(example)))
        cnt += 1
        # print(example['y_str'])
        # print(filter_pred_dict[annot_id]['cls_pred'], relabel_pred_dict[annot_id]['pred'])
        # print(' '.join(example['left_context_token']), '[[', example['mention_span'], ']]',
        #       ' '.join(example['right_context_token']))
        # print()
        # if i > 10:
        #     break
        if i % 100000 == 0:
            print(i)
    f.close()
    fout.close()
    print(cnt, 'examples written', nfcnt, 'not found')


data_dir = '/home/data/hldai/ultrafine'

# raw_data_file = os.path.join(data_dir, 'ld_data/train_full/open_train_tree_dhl_00.json')
# filter_pred_file = os.path.join(data_dir, 'ld_data/train_full/filter_eval_open_train_tree_dhl_00.json')
# relabel_pred_file = os.path.join(data_dir, 'ld_data/train_full/labeler_eval_open_train_tree_dhl_00.json')
# denoised_data_file = os.path.join(data_dir, 'ld_data/train_full/open_train_tree_dhl_00_denoised.json')
# __apply_denoise_results(raw_data_file, filter_pred_file, relabel_pred_file, denoised_data_file)

raw_data_file = os.path.join(data_dir, 'ld_data/train_full/el_train_full_tree_dhl.json')
filter_pred_file = os.path.join(data_dir, 'ld_data/train_full/filter_eval_el_train_full_tree_dhl.json')
relabel_pred_file = os.path.join(data_dir, 'ld_data/train_full/labeler_eval_el_train_full_tree_dhl.json')
denoised_data_file = os.path.join(data_dir, 'ld_data/train_full/el_train_full_tree_dhl_denoised.json')
__apply_denoise_results(raw_data_file, filter_pred_file, relabel_pred_file, denoised_data_file)
