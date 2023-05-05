import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
import tensorflow as tf

def plot_image(ax, image, title = '', grayscale=True, label=True, fontsize=None, save=False):
    im = ax.imshow(image < 0.5 if grayscale else image, aspect='auto', extent=(0, 384, 0, 9), cmap=plt.get_cmap('gray') if grayscale else None, vmin=0 if image.dtype == 'int8' else None, interpolation='none')
    if fontsize is not None:
        ax.set_title(title, fontsize=fontsize)
    else:
        ax.set_title(title)
    if label:
        ax.set_xlabel('η index')
        ax.set_ylabel('Layer index')
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    if save:
        plt.savefig(title + '.jpg', dpi=300, bbox_inches='tight')
    return im

def add_arrow(ax_parent, ax_child, xyA, xyB, color='black', lw=2.5):
    con = ConnectionPatch(xyA=xyA, xyB=xyB, coordsA='data', coordsB='data',
                          axesA=ax_child, axesB=ax_parent, arrowstyle='<|-, head_width=0.7, head_length=0.7',
                          color=color, linewidth=lw)
    ax_child.add_artist(con)

def plot_history(history):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for spread
    plt.plot(history.history['spread'])
    plt.plot(history.history['val_spread'])
    plt.title('model spread')
    plt.ylabel('spread')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for efficiency
    plt.plot(history.history['efficiency'])
    plt.plot(history.history['val_efficiency'])
    plt.title('model efficiency')
    plt.ylabel('efficiency')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    
def plot_cm(df: pd.DataFrame, norm=True, save=True):
    tp_index = []
    tn_index = []
    fp_index = []
    fn_index = []

    for i, elem in df.iterrows():
        # TP
        if elem['pt'] > 15 and elem['pt_pred'] > 15:
            tp_index.append(i)
        # TN
        elif elem['pt'] < 15 and elem['pt_pred'] < 15:
            tn_index.append(i)
        # FP
        elif elem['pt'] < 15 and elem['pt_pred'] > 15:
            fp_index.append(i)
        # FN
        else:
            fn_index.append(i)

    if norm:
        len_row1 = len(tp_index) + len(fn_index)
        len_row2 = len(fp_index) + len(tn_index)
        cfm = [[len(tp_index) / len_row1, len(fn_index) / len_row1],
               [len(fp_index) / len_row2, len(tn_index) / len_row2]]
    else:
        cfm = [[len(tp_index), len(fn_index)],
                [len(fp_index), len(tn_index)]]

    classes = ["$p_T$ > 15 GeV", "$p_T$ < 15 GeV"]
    columns = ["$\\hat{p}_T$ > 15 GeV", "$\\hat{p}_T$ < 15 GeV"]

    df_cfm = pd.DataFrame(cfm, index = classes, columns = columns)
    plt.figure(figsize = (8,6))
    cm_plot = sns.heatmap(df_cfm, annot=True, cmap='Blues', fmt='.2f' if norm else 'd', linewidths=0.5, linecolor='black', annot_kws={"size": 17})

    if save:
        cm_plot.figure.savefig('confusion_matrix.jpg', dpi=300, bbox_inches='tight')
        
    return tp_index, fn_index, fp_index, tn_index, cm_plot


def plot_logs(logs):
    # Loss
    plt.figure(figsize=(8,6))

    plt.plot(list(range(len(logs['train_loss_history']))), logs['train_loss_history'], label='Train loss')
    plt.plot(list(range(len(logs['valid_loss_history']))), logs['valid_loss_history'], label='Test loss')

    plt.title('Train vs Test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc="upper right")

    plt.show()

    # Spread
    plt.figure(figsize=(8,6))

    plt.plot(list(range(len(logs['train_spread_history']))), logs['train_spread_history'], label='Train spread')
    plt.plot(list(range(len(logs['valid_spread_history']))), logs['valid_spread_history'], label='Test spread')

    plt.title('Train vs Test spread')
    plt.xlabel('Epochs')
    plt.ylabel('Spread')
    plt.legend(loc="upper right")

    plt.show()

    # Efficiency
    plt.figure(figsize=(8,6))

    plt.plot(list(range(len(logs['train_efficiency_history']))), logs['train_efficiency_history'], label='Train efficiency')
    plt.plot(list(range(len(logs['valid_efficiency_history']))), logs['valid_efficiency_history'], label='Test efficiency')

    plt.title('Train vs Test efficiency')
    plt.xlabel('Epochs')
    plt.ylabel('Efficiency')
    plt.legend(loc="lower right")

    plt.show()

def plot_explanation(image, denoised_img, ram_eta, ram_pt, ig, sg, labels, predictions, index, save=False):
    # plotting procedure to show all the related images together, namely:
    # 1) image without noise
    # 2) image with noise
    # 3) RAM heatmap for eta
    # 4) RAM heatmap for pt
    # 5) IG heatmap
    # 6) SG heatmap
    # + all the overlappings
    plt.figure(figsize=(6.4*5, 4.8*2))
    
    # 1)
    plt.subplot(2, 5, 1)
    plt.imshow(denoised_img < 0.5, aspect='auto', extent=(0, 384, 0, 9), cmap=plt.get_cmap('gray'), vmin=0, interpolation='none')
    plt.title('Image without noise', fontsize=17)
    plt.xlabel('η index')
    plt.ylabel('Layer index')

    # 2)
    plt.subplot(2, 5, 6)
    plt.imshow(image < 0.5, aspect='auto', extent=(0, 384, 0, 9), cmap=plt.get_cmap('gray'), vmin=0, interpolation='none')
    plt.title('Image with noise', fontsize=17)
    plt.xlabel('η index')
    plt.ylabel('Layer index')

    # 3)
    plt.subplot(2, 5, 2)
    vmin = ram_eta.min()
    vmax = ram_eta.max()
    norm_eta = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    plt.imshow(ram_eta, aspect='auto', extent=(0, 384, 0, 9), cmap='coolwarm', norm=norm_eta, interpolation='none')
    plt.colorbar().set_ticks([])
    plt.title('RAM for feature $\\eta$', fontsize=17)
    plt.xlabel('η index')
    plt.ylabel('Layer index')

    # 4)
    plt.subplot(2, 5, 3)
    vmin = ram_pt.min()
    vmax = ram_pt.max()
    norm_pt = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    plt.imshow(ram_pt, aspect='auto', extent=(0, 384, 0, 9), cmap='coolwarm', norm=norm_pt, interpolation='none')
    plt.colorbar().set_ticks([])
    plt.title('RAM for feature $p_T$', fontsize=17)
    plt.xlabel('η index')
    plt.ylabel('Layer index')

    # 5)
    plt.subplot(2, 5, 4)
    vmin = tf.reduce_min(ig)
    vmax = tf.reduce_max(ig)
    norm_ig = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    plt.imshow(ig, aspect='auto', extent=(0, 384, 0, 9), cmap='coolwarm', norm=norm_ig, interpolation='none')
    plt.colorbar().set_ticks([])
    plt.title('Integrated Gradients', fontsize=17)
    plt.xlabel('η index')
    plt.ylabel('Layer index')

    # 6)
    plt.subplot(2, 5, 5)
    vmin = tf.reduce_min(sg)
    vmax = tf.reduce_max(sg)
    norm_sg = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    plt.imshow(sg, aspect='auto', extent=(0, 384, 0, 9), cmap='coolwarm', norm=norm_sg, interpolation='none')
    plt.colorbar().set_ticks([])
    plt.title('SmoothGrad', fontsize=17)
    plt.xlabel('η index')
    plt.ylabel('Layer index')

    # overlappings
    plt.subplot(2, 5, 7)
    plt.imshow(image < 0.5, aspect='auto', extent=(0, 384, 0, 9), cmap=plt.get_cmap('gray'), alpha=0.5, interpolation='none')
    plt.imshow(ram_eta, aspect='auto', extent=(0, 384, 0, 9), cmap='coolwarm', alpha=0.5, norm=norm_eta,interpolation='none')
    plt.colorbar().set_ticks([])
    plt.title('Superimposed image and RAM - $\\eta$', fontsize=17)
    plt.xlabel('η index')
    plt.ylabel('Layer index')

    plt.subplot(2, 5, 8)
    plt.imshow(image < 0.5, aspect='auto', extent=(0, 384, 0, 9), cmap=plt.get_cmap('gray'), alpha=0.5, interpolation='none')
    plt.imshow(ram_pt, aspect='auto', extent=(0, 384, 0, 9), cmap='coolwarm', alpha=0.5, norm=norm_pt, interpolation='none')
    plt.colorbar().set_ticks([])
    plt.title('Superimposed image and RAM - $p_T$', fontsize=17)
    plt.xlabel('η index')
    plt.ylabel('Layer index')

    plt.subplot(2, 5, 9)
    plt.imshow(image < 0.5, aspect='auto', extent=(0, 384, 0, 9), cmap=plt.get_cmap('gray'), alpha=0.5, interpolation='none')
    plt.imshow(ig, aspect='auto', extent=(0, 384, 0, 9), cmap='coolwarm', alpha=0.5, norm=norm_ig, interpolation='none')
    plt.colorbar().set_ticks([])
    plt.title('Superimposed image and IG', fontsize=17)
    plt.xlabel('η index')
    plt.ylabel('Layer index')

    plt.subplot(2, 5, 10)
    plt.imshow(image < 0.5, aspect='auto', extent=(0, 384, 0, 9), cmap=plt.get_cmap('gray'), alpha=0.5, interpolation='none')
    plt.imshow(sg, aspect='auto', extent=(0, 384, 0, 9), cmap='coolwarm', alpha=0.5, norm=norm_sg, interpolation='none')
    plt.colorbar().set_ticks([])
    plt.title('Superimposed image and SG', fontsize=17)
    plt.xlabel('η index')
    plt.ylabel('Layer index')

    plt.suptitle(f'Real: [$p_T$={labels[0]:.1f} GeV, $\\eta$={labels[1]:.2f}] Predicted: [$p_T$={predictions[0]:.1f} GeV, $\\eta$={predictions[1]:.2f}]', fontsize=25)

    if save:
        plt.savefig('attribution_' + str(index) + '.jpg', dpi=300, bbox_inches='tight')

    plt.show()