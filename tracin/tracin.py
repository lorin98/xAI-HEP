import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.plot import plot_image

@tf.function
def get_loss_grads(models_penultimate, models_last, inputs):
    imageids, images, labels = inputs
    loss_grads = []
    activations = []
    for mp, ml in zip(models_penultimate, models_last):
        h = mp(images)
        preds = ml(h)
        loss_grad = labels - preds
        loss_grads.append(loss_grad)
        activations.append(h)

    # preds, _ = tf.math.top_k(preds, k=2)

    return imageids, tf.stack(loss_grads, axis=-1), tf.stack(activations, axis=-1), labels, preds


def get_trackin_grad(models_penultimate, models_last, images, ground_truth, batch_size):
    image_ids_np = []
    loss_grads_np = []
    activations_np = []
    labels_np = []
    preds_np = []
    
    for i in tqdm(range(0, images.shape[0], batch_size), desc='Batch TrackIn'):
        batch = images[i:i+batch_size, :, :]
        batch_labels = tf.constant(ground_truth[i:i+batch_size, :2], dtype=tf.float32)
        imageids = np.array([id for id in range(i, i+batch_size)])
        # print(imageids)
        # print(imageids.shape)
        # print(batch)
        # print(batch.shape)
        # print(batch_labels)
        # print(batch_labels.shape)

        imageids, loss_grads, activations, labels, preds = get_loss_grads(models_penultimate, models_last, (imageids, batch, batch_labels))
        
        image_ids_np.append(imageids.numpy())
        loss_grads_np.append(loss_grads.numpy())
        activations_np.append(activations.numpy())
        labels_np.append(labels.numpy())
        preds_np.append(preds.numpy())

    return {'image_ids': np.concatenate(image_ids_np),
            'loss_grads': np.concatenate(loss_grads_np),
            'activations': np.concatenate(activations_np),
            'labels': np.concatenate(labels_np),
            'preds': np.concatenate(preds_np)
            }

def find_prop_opp(trackin_train,  # list of dictionaries for the training samples
                  trackin_test,   # list of dictionaries for the test samples
                  idx,            # indices of the test sample we want to explain
                  train_images,   
                  train_images_without_noise,
                  test_images,
                  test_images_without_noise,
                  topk=3,          # number of proponents/opponents to show
                  title=None
                  ):

    scores = []
    scores_lg = []
    scores_a = []
    loss_grad = trackin_test['loss_grads'][idx]
    activation = trackin_test['activations'][idx]
    
    for i in range(len(trackin_train['loss_grads'])):
        lg_sim = np.sum(trackin_train['loss_grads'][i] * loss_grad, axis=0)
        a_sim = np.sum(trackin_train['activations'][i] * activation, axis=0)
        scores.append(np.sum(lg_sim * a_sim))
        scores_lg.append(np.sum(lg_sim))
        scores_a.append(np.sum(a_sim))
        
    proponents = []
    opponents = []
    indices = np.argsort(scores)
    #sns.displot(scores, kind='kde')

    for i in range(topk):
        index = indices[-i-1]
        proponents.append(
            (
                trackin_train['image_ids'][index],
                trackin_train['preds'][index],
                trackin_train['labels'][index],
                scores[index],
                scores_lg[index] if scores_lg else "Not Computed",
                scores_a[index] if scores_a else "Not Computed",
                index
            )
        )

        index = indices[i]
        opponents.append(
            (
                trackin_train['image_ids'][index],
                trackin_train['preds'][index],
                trackin_train['labels'][index],
                scores[index],
                scores_lg[index] if scores_lg else "Not Computed",
                scores_a[index] if scores_a else "Not Computed",
                index
            )
        )
    
    if topk == 3:
        fig = plt.figure(figsize=(6.4*4, 4.8*2))

        print(f'Test image [{idx}]:')
        ax = plt.subplot(2, 4, 1)
        pos = ax.get_position()
        new_pos = [pos.x0 - 0.05, pos.y0 - 0.3, pos.width, pos.height]
        ax.set_position(new_pos)
        if test_images_without_noise is not None:
            rows, cols = np.where(test_images_without_noise[idx] != 0)
        I = np.dstack([test_images[idx], test_images[idx], test_images[idx]]).astype(np.int16)
        I[np.where(I == 0)[0], np.where(I == 0)[1], :] = [255, 255, 255]
        if test_images_without_noise is not None:
            I[rows, cols, :] = [255, 0, 0]
        plot_image(ax, I, title=f'Input image\n\nReal: [$p_T$={trackin_test["labels"][idx][0]:.1f} GeV, $\\eta$={trackin_test["labels"][idx][1]:.2f}]\nPredicted: [$p_T$={trackin_test["preds"][idx][0]:.1f} GeV, $\\eta$={trackin_test["preds"][idx][1]:.2f}]', grayscale=False, fontsize=20)
        # ax = plt.subplot(2, 4, 5)
        # if test_images_without_noise is not None:
        #     plot_image(ax, test_images_without_noise[idx], title='Denoised image\n\n\n', fontsize=20)
        # else:
        #     denoised_input_img = np.zeros((9, 384))
        #     denoised_input_img[0][0] = 1
        #     plot_image(ax, denoised_input_img, title='Denoised image\n\n\n', fontsize=20)

        idxs_prop = []
        for p, prop in enumerate(proponents):
            idxs_prop.append(prop[6])
            ax = plt.subplot(2, 4, p+2)
            rows, cols = np.where(train_images_without_noise[prop[0]] != 0)
            I = np.dstack([train_images[prop[0]], train_images[prop[0]], train_images[prop[0]]]).astype(np.int16)
            I[np.where(I == 0)[0], np.where(I == 0)[1], :] = [255, 255, 255]
            I[rows, cols, :] = [255, 0, 0]

            if p == 1:
                tit = f'Proponents\n\nReal: [$p_T$={prop[2][0]:.1f} GeV, $\\eta$={prop[2][1]:.2f}]\nPredicted: [$p_T$={prop[1][0]:.1f} GeV, $\\eta$={prop[1][1]:.2f}]' # \nInfluence: {prop[3]:.1f}'
            else:
                tit = f'Real: [$p_T$={prop[2][0]:.1f} GeV, $\\eta$={prop[2][1]:.2f}]\nPredicted: [$p_T$={prop[1][0]:.1f} GeV, $\\eta$={prop[1][1]:.2f}]' # \nInfluence: {prop[3]:.1f}'
            plot_image(ax,
                    I,
                    title=tit,
                    grayscale=False,
                    fontsize=20)

        idxs_opp = []
        for o, opp in enumerate(opponents):
            idxs_opp.append(opp[6])
            ax = plt.subplot(2, 4, o+6)
            pos = ax.get_position()
            new_pos = [pos.x0, pos.y0 - 0.2, pos.width, pos.height]
            ax.set_position(new_pos)
            rows, cols = np.where(train_images_without_noise[opp[0]] != 0)
            I = np.dstack([train_images[opp[0]], train_images[opp[0]], train_images[opp[0]]]).astype(np.int16)
            I[np.where(I == 0)[0], np.where(I == 0)[1], :] = [255, 255, 255]
            I[rows, cols, :] = [255, 0, 0]

            if o == 1:
                tit = f'Opponents\n\nReal: [$p_T$={opp[2][0]:.1f} GeV, $\\eta$={opp[2][1]:.2f}]\nPredicted: [$p_T$={opp[1][0]:.1f} GeV, $\\eta$={opp[1][1]:.2f}]' # \nInfluence: {opp[3]:.1f}'
            else:
                tit = f'Real: [$p_T$={opp[2][0]:.1f} GeV, $\\eta$={opp[2][1]:.2f}]\nPredicted: [$p_T$={opp[1][0]:.1f} GeV, $\\eta$={opp[1][1]:.2f}]' # \nInfluence: {opp[3]:.1f}'
            plot_image(ax,
                    I,
                    title=tit,
                    grayscale=False,
                    fontsize=20)

        # line = plt.Line2D([0.305, 0.305], [0, 1], transform=fig.transFigure, color="black")
        # fig.add_artist(line)
        # line = plt.Line2D([0.305, 0.9], [0.505, 0.505], transform=fig.transFigure, color="black")
        # fig.add_artist(line)
        # plt.subplots_adjust(hspace=0.8)

        if title is not None:
            plt.savefig(title, dpi=300, bbox_inches='tight')

        plt.show()

        print('Proponents indices: ', idxs_prop)
        print('Opponents indices: ', idxs_opp)
    
    else:
        print(f'The visualization is designed for topk=3. You have specified topk={topk}.')