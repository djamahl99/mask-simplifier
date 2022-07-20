import json
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    labels_txt = open("annotations/labels.txt").read()
    labels = labels_txt.splitlines()


    j = json.loads(open("annotations/instances_val2017.json").read())

    category_stats = {}

    print(j['info'])

    # print([i for i in j['annotations']])
    exit()
    annots = j['annotations']
    # print([a['category_id'] for a in annot[0:500]])

    for annot in annots:
        cat_id = annot['category_id']
        if cat_id not in category_stats:
            category_stats[cat_id] = {
                'num_segments': [],
                'area': []
            }

        # print(annot['segmentation'][0], len(annot['segmentation'][0]))

        try:
            annot['segmentation'][0]
        except:
            print("annot", annot)
            # exit()

        if 'counts' in annot['segmentation']:
            category_stats[cat_id]['num_segments'].append(len(annot['segmentation']['counts']) // 2)
        else:
            category_stats[cat_id]['num_segments'].append(len(annot['segmentation'][0]) // 2)
            
        category_stats[cat_id]['area'].append(annot['area'])

    m_ = 0

    for cat_id in category_stats:
        plt.hist(category_stats[cat_id]['num_segments'], 100, density=True)
        print(cat_id, labels[cat_id - 1])
        plt.title(labels[cat_id - 1])

        m = np.mean(np.array(category_stats[cat_id]['num_segments']))
        m_ = m if m > m_ else m_

    print("largest mean", m_)
    plt.show()

    #     category_stats[cat_id]['num_segments'] = np.mean(np.array(category_stats[cat_id]['num_segments']))
    #     category_stats[cat_id]['area'] = np.mean(np.array(category_stats[cat_id]['area']))

    # print(category_stats)