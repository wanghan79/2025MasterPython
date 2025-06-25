from collections import defaultdict
from utils import cosine_similarity

class UserCF:
    def __init__(self, train_data):
        self.train_data = self._build_user_item(train_data)
        self.user_sim = self._calc_user_sim()

    def _build_user_item(self, data):
        user_item = defaultdict(dict)
        for _, row in data.iterrows():
            user_item[row['user_id']][row['item_id']] = row['rating']
        return user_item

    def _calc_user_sim(self):
        sim = {}
        users = list(self.train_data.keys())
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                u, v = users[i], users[j]
                score = cosine_similarity(self.train_data[u], self.train_data[v])
                sim.setdefault(u, {})[v] = score
                sim.setdefault(v, {})[u] = score
        return sim

    def recommend(self, user_id, k=10):
        if user_id not in self.user_sim:
            return []
        sim_users = sorted(self.user_sim[user_id].items(), key=lambda x: x[1], reverse=True)[:10]
        scores = defaultdict(float)
        for sim_user, sim_score in sim_users:
            for item in self.train_data[sim_user]:
                if item not in self.train_data[user_id]:
                    scores[item] += sim_score * self.train_data[sim_user][item]
        ranked_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, _ in ranked_items[:k]]