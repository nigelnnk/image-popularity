from sklearn.model_selection import train_test_split


def split_dataset(
        dataset,
        train_size,
        val_size,
        stratify_by=None,
        random_seed=0):
    if stratify_by:
        stratify = dataset[stratify_by]
    else:
        stratify = None
    train, val_test = train_test_split(
        dataset,
        train_size=train_size,
        random_state=random_seed,
        shuffle=True,
        stratify=stratify)

    if stratify_by:
        stratify = val_test[stratify_by]
    else:
        stratify = None
    val, test = train_test_split(
        val_test,
        train_size=val_size / (1.0 - train_size),
        random_state=random_seed,
        shuffle=True,
        stratify=stratify)

    return train, val, test
