"""Step 3: train the multilayer perceptron and evaluate it on a held-out test set."""
import argparse
import os
import sys
import time

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from reaction_feasibility.compress import CompressShelve  # noqa: E402
from reaction_feasibility.model import batch_generator, build_mlp, n_descriptors_in  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shelve", default="descriptors.shelve", help="descriptor store from step 2")
    p.add_argument("--hidden", type=int, default=1000, help="neurons in the hidden layer")
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--best-model", default="best_model.h5", help="checkpoint of the best epoch")
    p.add_argument("--final-model", default="final_model.h5")
    p.add_argument("--logdir", default="logs")
    a = p.parse_args()

    bs = a.batch_size
    with CompressShelve(a.shelve, flag="r") as f:
        indices = list(f.keys())
        n_desc = n_descriptors_in(f)

        # 90 % train+validation / 10 % test, then 80 / 20 train / validation.
        pretrain, test = train_test_split(indices, test_size=0.1, random_state=42, shuffle=True)
        train, validation = train_test_split(pretrain, test_size=0.2, random_state=41)
        print(f"train {len(train)}, validation {len(validation)}, test {len(test)}")

        run_name = f"mlp_{n_desc}_descriptors_{int(time.time())}"
        callbacks = [
            EarlyStopping(monitor="val_balanced_acc", mode="max", patience=3),
            ReduceLROnPlateau(monitor="val_balanced_acc", mode="max", factor=0.5, patience=1,
                              min_lr=1e-6, min_delta=1e-5, verbose=1),
            ModelCheckpoint(a.best_model, save_best_only=True, monitor="val_balanced_acc", mode="max"),
            TensorBoard(log_dir=os.path.join(a.logdir, run_name), histogram_freq=1),
        ]

        model = build_mlp(n_desc, a.hidden)
        model.fit(batch_generator(train, bs, f),
                  steps_per_epoch=len(train) // bs,
                  validation_data=batch_generator(validation, bs, f),
                  validation_steps=len(validation) // bs,
                  epochs=a.epochs, verbose=1, callbacks=callbacks)
        model.save(a.final_model)

        results = model.evaluate(batch_generator(test, bs, f),
                                 steps=max(1, len(test) // bs), verbose=1, return_dict=True)
        print("Test set:", results)


if __name__ == "__main__":
    main()
