from data_loader import load_data, train_test_split
from user_cf import UserCF
from evaluate import precision_at_k, recall_at_k

def get_user_truth(test_data):
    truth = {}
    for _, row in test_data.iterrows():
        truth.setdefault(row['user_id'], set()).add(row['item_id'])
    return truth

if __name__ == "__main__":
    data = load_data()
    train, test = train_test_split(data)
    model = UserCF(train)

    truth = get_user_truth(test)
    prec, rec = [], []
    for user_id in truth:
        recs = model.recommend(user_id, k=10)
        if recs:
            prec.append(precision_at_k(recs, truth[user_id]))
            rec.append(recall_at_k(recs, truth[user_id]))
    print(f"Precision@10: {sum(prec)/len(prec):.4f}")
    print(f"Recall@10: {sum(rec)/len(rec):.4f}")