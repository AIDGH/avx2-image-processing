import os, struct
from PIL import Image

def write_idx_images(path, images, rows=28, cols=28):
    with open(path, "wb") as f:
        f.write(struct.pack(">IIII", 2051, len(images), rows, cols))
        for img in images:
            f.write(img)

def write_idx_labels(path, labels):
    with open(path, "wb") as f:
        f.write(struct.pack(">II", 2049, len(labels)))
        f.write(bytes(labels))

def load_split(root):  # root مثل train یا test
    images = []
    labels = []
    for d in range(10):
        folder = os.path.join(root, str(d))
        files = sorted([x for x in os.listdir(folder) if x.lower().endswith(".png")])
        for fn in files:
            p = os.path.join(folder, fn)
            im = Image.open(p).convert("L").resize((28,28))
            images.append(im.tobytes())
            labels.append(d)
    return images, labels

if __name__ == "__main__":
    base = "mnist-pngs-main"
    tr_imgs, tr_lbls = load_split(os.path.join(base, "train"))
    te_imgs, te_lbls = load_split(os.path.join(base, "test"))

    os.makedirs("mnist_idx", exist_ok=True)
    write_idx_images("mnist_idx/train-images-idx3-ubyte", tr_imgs)
    write_idx_labels("mnist_idx/train-labels-idx1-ubyte", tr_lbls)
    write_idx_images("mnist_idx/t10k-images-idx3-ubyte", te_imgs)
    write_idx_labels("mnist_idx/t10k-labels-idx1-ubyte", te_lbls)

    print("done:", len(tr_imgs), len(te_imgs))
