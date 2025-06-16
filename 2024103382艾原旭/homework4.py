#ICEWS05-15/ICEWS14数据集预处理核心代码
def prepare_dataset(path, name):
    files = ['train', 'valid', 'test']
    entities, relations, timestamps = set(), set(), set()
    for f in files:
        file_path = os.path.join(path, f)
        to_read = open(file_path, 'r')
        for line in to_read.readlines():
            lhs, rel, rhs, timestamp = line.strip().split('\t')
            entities.add(lhs)
            entities.add(rhs)
            relations.add(rel)
            timestamps.add(timestamp)
        to_read.close()

    entities_to_id = {x: i for (i, x) in enumerate(sorted(entities))}
    relations_to_id = {x: i for (i, x) in enumerate(sorted(relations))}
    timestamps_to_id = {x: i for (i, x) in enumerate(sorted(timestamps))}

    print("{} entities, {} relations over {} timestamps".format(len(entities), len(relations), len(timestamps)))
    n_relations = len(relations)
    n_entities = len(entities)

    os.makedirs(os.path.join(DATA_PATH, name))
    # write ent to id / rel to id
    for (dic, f) in zip([entities_to_id, relations_to_id, timestamps_to_id], ['ent_id', 'rel_id', 'ts_id']):
        ff = open(os.path.join(DATA_PATH, name, f), 'w+')
        for (x, i) in dic.items():
            ff.write("{}\t{}\n".format(x, i))
        ff.close()

    # map train/test/valid with the ids
    for f in files:
        file_path = os.path.join(path, f)
        to_read = open(file_path, 'r')
        examples = []
        for line in to_read.readlines():
            lhs, rel, rhs, ts = line.strip().split('\t')
            try:
                examples.append([entities_to_id[lhs], relations_to_id[rel], entities_to_id[rhs], timestamps_to_id[ts]])
            except ValueError:
                continue
        out = open(Path(DATA_PATH) / name / (f + '.pickle'), 'wb')
        pickle.dump(np.array(examples).astype('uint64'), out)
        out.close()

    print("creating filtering lists")

    # create filtering files
    to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}
    for f in files:
        examples = pickle.load(open(Path(DATA_PATH) / name / (f + '.pickle'), 'rb'))
        for lhs, rel, rhs, ts in examples:
            to_skip['lhs'][(rhs, rel + n_relations, ts)].add(lhs)  # reciprocals
            to_skip['rhs'][(lhs, rel, ts)].add(rhs)

    to_skip_final = {'lhs': {}, 'rhs': {}}
    for kk, skip in to_skip.items():
        for k, v in skip.items():
            to_skip_final[kk][k] = sorted(list(v))

    out = open(Path(DATA_PATH) / name / 'to_skip.pickle', 'wb')
    pickle.dump(to_skip_final, out)
    out.close()

    examples = pickle.load(open(Path(DATA_PATH) / name / 'train.pickle', 'rb'))
    counters = {
        'lhs': np.zeros(n_entities),
        'rhs': np.zeros(n_entities),
        'both': np.zeros(n_entities)
    }

    for lhs, rel, rhs, _ts in examples:
        counters['lhs'][lhs] += 1
        counters['rhs'][rhs] += 1
        counters['both'][lhs] += 1
        counters['both'][rhs] += 1
    for k, v in counters.items():
        counters[k] = v / np.sum(v)
    out = open(Path(DATA_PATH) / name / 'probas.pickle', 'wb')
    pickle.dump(counters, out)
    out.close()

#wiki12K数据集预处理核心代码
def get_be(begin, end):
    begin = begin.strip().split('-')[0]
    end = end.strip().split('-')[0]
    if begin =='####':
        begin = (-math.inf, 0, 0)
    else:
        begin = (int(begin), 0, 0)
    if end == '####':
        end = (math.inf, 0, 0)
    else:
        end = (int(end), 0, 0)

    return begin, end


def prepare_dataset_rels(path, name):
    """
    Given a path to a folder containing tab separated files :
     train, test, valid
    In the format :
    (lhs)\t(rel)\t(rhs)\t(type)\t(timestamp)\n
    Maps each entity, relation+type and timestamp to a unique id, create corresponding folder
    name in pkg/data, with mapped train/test/valid files.
    Also create to_skip_lhs / to_skip_rhs for filtered metrics and
    rel_id / ent_id for analysis.
    """
    files = ['train', 'valid', 'test']
    entities, relations, timestamps = set(), set(), set()
    for f in files:
        file_path = os.path.join(path, f)
        to_read = open(file_path, 'r')
        for line in to_read.readlines():
            v = line.strip().split('\t')
            lhs, rel, rhs, begin, end = v

            begin, end = get_be(begin, end)

            timestamps.add(begin)
            timestamps.add(end)
            entities.add(lhs)
            entities.add(rhs)
            relations.add(rel)

        to_read.close()

    print(f"{len(timestamps)} timestamps")

    entities_to_id = {x: i for (i, x) in enumerate(sorted(entities))}
    relations_to_id = {x: i for (i, x) in enumerate(sorted(relations))}

    # we need to sort timestamps and associate them to actual dates
    all_ts = sorted(timestamps)[1:-1]
    timestamps_to_id = {x: i for (i, x) in enumerate(all_ts)}
    # print(timestamps_to_id)

    print("{} entities, {} relations over {} timestamps".format(len(entities), len(relations), len(timestamps)))
    n_relations = len(relations)
    n_entities = len(entities)

    try:
        os.makedirs(os.path.join(DATA_PATH, name))
    except OSError as e:
        r = input(f"{e}\nContinue ? [y/n]")
        if r != "y":
            sys.exit()

    # write ent to id / rel to id
    for (dic, f) in zip([entities_to_id, relations_to_id, timestamps_to_id], ['ent_id', 'rel_id', 'ts_id']):
        ff = open(os.path.join(DATA_PATH, name, f), 'wb')
        pickle.dump(dic, ff)
        ff.close()

    # dump the time differences between timestamps for continuity regularizer
    # ignores number of days in a month but who cares
    # ts_to_int = [x[0] * 365 + x[1] * 30 + x[2] for x in all_ts]
    ts_to_int = [x[0] for x in all_ts]
    ts = np.array(ts_to_int, dtype='float')
    diffs = ts[1:] - ts[:-1]  # remove last timestamp from time diffs. it's not really a timestamp
    out = open(os.path.join(DATA_PATH, name, 'ts_diffs.pickle'), 'wb')
    pickle.dump(diffs, out)
    out.close()

    # map train/test/valid with the ids
    event_list = {
        'all': [],
    }
    for f in files:
        file_path = os.path.join(path, f)
        to_read = open(file_path, 'r')
        examples = []
        ignore = 0
        total = 0
        full_intervals = 0
        half_intervals = 0
        point = 0
        for line in to_read.readlines():
            v = line.strip().split('\t')
            lhs, rel, rhs, begin, end = v
            begin_t, end_t = get_be(begin, end)
            total += 1

            begin = begin_t
            end = end_t

            if begin_t[0] == -math.inf:
                begin = all_ts[0]
                if not end_t[0] == math.inf:
                    half_intervals += 1
            if end_t[0] == math.inf:
                end = all_ts[-1]
                if not begin_t[0] == -math.inf:
                    half_intervals += 1

            if begin_t[0] > -math.inf and end_t[0] < math.inf:
                if begin_t[0] == end_t[0]:
                    point += 1
                else:
                    full_intervals += 1

            begin = timestamps_to_id[begin]
            end = timestamps_to_id[end]

            if begin > end:
                ignore += 1
                continue

            lhs = entities_to_id[lhs]
            rel = relations_to_id[rel]
            rhs = entities_to_id[rhs]

            event_list['all'].append((begin, -1, (lhs, rel, rhs)))
            event_list['all'].append((end, +1, (lhs, rel, rhs)))

            try:
                examples.append([lhs, rel, rhs, begin, end])
            except ValueError:
                continue
        out = open(Path(DATA_PATH) / name / (f + '.pickle'), 'wb')
        pickle.dump(np.array(examples).astype('uint64'), out)
        out.close()
        print(f"Ignored {ignore} events.")
        print(f"Total : {total} // Full : {full_intervals} // Half : {half_intervals} // Point : {point}")

    for k, v in event_list.items():
        out = open(Path(DATA_PATH) / name / ('event_list_' + k + '.pickle'), 'wb')
        print("Dumping all events", len(v))
        pickle.dump(sorted(v), out)
        out.close()

#训练部分核心代码
class LCGE(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int, Rules, w_static,
            no_time_emb=False, init_size: float = 1e-3
    ):
        super(LCGE, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.rank_static = rank // 20
        self.w_static = w_static

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in [sizes[0], sizes[1], sizes[3], sizes[1], 1]  # last embedding modules contains no_time embeddings
        ])
        self.static_embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * self.rank_static, sparse=True)
            for s in [sizes[0], sizes[1]]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size
        self.embeddings[3].weight.data *= init_size
        self.embeddings[4].weight.data *= init_size  # time transition
        self.static_embeddings[0].weight.data *= init_size  # static entity embedding
        self.static_embeddings[1].weight.data *= init_size  # static relation embedding

        self.no_time_emb = no_time_emb
        # self.rule1_p1, self.rule1_p2, self.rule2_p1, self.rule2_p2, self.rule2_p3, self.rule2_p4 = Rules
        self.rule1_p1, self.rule1_p2, self.rule2_p1, self.rule2_p2, self.rule2_p3, self.rule2_p4 = Rules

    @staticmethod
    def has_time():
        return True

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rel_no_time = self.embeddings[3](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]
        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = (rt[0] - rt[3]) + rnt[0], (rt[1] + rt[2]) + rnt[1]

        h_static = self.static_embeddings[0](x[:, 0])
        r_static = self.static_embeddings[1](x[:, 1])
        t_static = self.static_embeddings[0](x[:, 2])

        h_static = h_static[:, :self.rank_static], h_static[:, self.rank_static:]
        r_static = r_static[:, :self.rank_static], r_static[:, self.rank_static:]
        t_static = t_static[:, :self.rank_static], t_static[:, self.rank_static:]
        # print("h size:{}\tr size:{}\ttsize:{}".format(h_static[0].shape, r_static[0].shape, t_static[0].shape))

        return torch.sum(
            (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) * rhs[0] +
            (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) * rhs[1],
            1, keepdim=True
        ), torch.sum(
            (h_static[0] * r_static[0] - h_static[1] * r_static[1]) * t_static[0] +
            (h_static[1] * r_static[0] + h_static[0] * r_static[1]) * t_static[1],
            1, keepdim=True
        )

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rel_no_time = self.embeddings[3](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])
        transt = self.embeddings[4](torch.LongTensor([0]).cuda())

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]
        transt = transt[:, :self.rank], transt[:, self.rank:]
        # print(transt[1])

        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        rrt = rt[0] - rt[3], rt[1] + rt[2]
        full_rel = rrt[0] + rnt[0], rrt[1] + rnt[1]

        h_static = self.static_embeddings[0](x[:, 0])
        r_static = self.static_embeddings[1](x[:, 1])
        t_static = self.static_embeddings[0](x[:, 2])

        h_static = h_static[:, :self.rank_static], h_static[:, self.rank_static:]
        r_static = r_static[:, :self.rank_static], r_static[:, self.rank_static:]
        t_static = t_static[:, :self.rank_static], t_static[:, self.rank_static:]

        right_static = self.static_embeddings[0].weight
        right_static = right_static[:, :self.rank_static], right_static[:, self.rank_static:]

        regularizer = (
            math.pow(2, 1 / 3) * torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
            torch.sqrt(rrt[0] ** 2 + rrt[1] ** 2),
            torch.sqrt(rnt[0] ** 2 + rnt[1] ** 2),
            math.pow(2, 1 / 3) * torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2),
            torch.sqrt(h_static[0] ** 2 + h_static[1] ** 2),
            torch.sqrt(r_static[0] ** 2 + r_static[1] ** 2),
            torch.sqrt(t_static[0] ** 2 + t_static[1] ** 2)
        )

        rule = 0.
        rule_num = 0
        for rel_1 in x[:, 1]:
            rel_1_str = str(rel_1.item())
            if rel_1_str in self.rule1_p2:
                rel1_emb = self.embeddings[3](rel_1)
                for rel_2 in self.rule1_p2[rel_1_str]:
                    weight_r = self.rule1_p2[rel_1_str][rel_2]
                    rel2_emb = self.embeddings[3](torch.LongTensor([int(rel_2)]).cuda())[0]
                    rule += weight_r * torch.sum(torch.abs(rel1_emb - rel2_emb) ** 3)
                    rule_num += 1

        for rel_1 in x[:, 1]:
            rel_1_str = str(rel_1.item())
            if rel_1_str in self.rule1_p2:
                rel1_emb = self.embeddings[3](rel_1)
                rel1_split = rel1_emb[:self.rank], rel1_emb[self.rank:]
                for rel_2 in self.rule1_p2[rel_1_str]:
                    weight_r = self.rule1_p2[rel_1_str][rel_2]
                    rel2_emb = self.embeddings[3](torch.LongTensor([int(rel_2)]).cuda())[0]
                    rel2_split = rel2_emb[:self.rank], rel2_emb[self.rank:]
                    tt = rel2_split[0] * transt[0][0], rel2_split[1] * transt[0][0], rel2_split[0] * transt[1][0], \
                         rel2_split[1] * transt[1][0]
                    rtt = tt[0] - tt[3], tt[1] + tt[2]
                    # print("rel1_split:\t", rel1_split[0])
                    rule += weight_r * (torch.sum(torch.abs(rel1_split[0] - rtt[0]) ** 3) + torch.sum(
                        torch.abs(rel1_split[1] - rtt[1]) ** 3))
                    rule_num += 1

        for rel_1 in x[:, 1]:
            if rel_1 in self.rule2_p1:
                rel1_emb = self.embeddings[3](rel_1)
                rel1_split = rel1_emb[:self.rank], rel1_emb[self.rank:]
                for body in self.rule2_p1[rel_1]:
                    rel_2, rel_3 = body
                    weight_r = self.rule2_p1[rel_1][body]
                    rel2_emb = self.embeddings[3](torch.LongTensor([rel_2]).cuda())[0]
                    rel3_emb = self.embeddings[3](torch.LongTensor([rel_3]).cuda())[0]
                    rel2_split = rel2_emb[:self.rank], rel2_emb[self.rank:]
                    rel3_split = rel3_emb[:self.rank], rel3_emb[self.rank:]
                    tt2 = rel2_split[0] * transt[0][0], rel2_split[1] * transt[0][0], rel2_split[0] * transt[1][0], \
                          rel2_split[1] * transt[1][0]
                    rtt2 = tt2[0] - tt2[3], tt2[1] + tt2[2]
                    ttt2 = rtt2[0] * transt[0][0], rtt2[1] * transt[0][0], rtt2[0] * transt[1][0], rtt2[1] * transt[1][
                        0]
                    rttt2 = ttt2[0] - ttt2[3], ttt2[1] + ttt2[2]
                    tt3 = rel3_split[0] * transt[0][0], rel3_split[1] * transt[0][0], rel3_split[0] * transt[1][0], \
                          rel3_split[1] * transt[1][0]
                    rtt3 = tt3[0] - tt3[3], tt3[1] + tt3[2]
                    tt = rtt3[0] * rttt2[0], rtt3[1] * rttt2[0], rtt3[0] * rttt2[1], rtt3[1] * rttt2[1]
                    rtt = tt[0] - tt[3], tt[1] + tt[2]
                    # print("rel1_split:\t", rel1_split[0])
                    rule += weight_r * (torch.sum(torch.abs(rel1_split[0] - rtt[0]) ** 3) + torch.sum(
                        torch.abs(rel1_split[1] - rtt[1]) ** 3))
                    rule_num += 1

        for rel_1 in x[:, 1]:
            if rel_1 in self.rule2_p2:
                rel1_emb = self.embeddings[3](rel_1)
                rel1_split = rel1_emb[:self.rank], rel1_emb[self.rank:]
                for body in self.rule2_p2[rel_1]:
                    rel_2, rel_3 = body
                    weight_r = self.rule2_p2[rel_1][body]
                    rel2_emb = self.embeddings[3](torch.LongTensor([rel_2]).cuda())[0]
                    rel3_emb = self.embeddings[3](torch.LongTensor([rel_3]).cuda())[0]
                    rel2_split = rel2_emb[:self.rank], rel2_emb[self.rank:]
                    rel3_split = rel3_emb[:self.rank], rel3_emb[self.rank:]
                    tt2 = rel2_split[0] * transt[0][0], rel2_split[1] * transt[0][0], rel2_split[0] * transt[1][0], \
                          rel2_split[1] * transt[1][0]
                    rtt2 = tt2[0] - tt2[3], tt2[1] + tt2[2]
                    tt3 = rel3_split[0] * transt[0][0], rel3_split[1] * transt[0][0], rel3_split[0] * transt[1][0], \
                          rel3_split[1] * transt[1][0]
                    rtt3 = tt3[0] - tt3[3], tt3[1] + tt3[2]
                    tt = rtt3[0] * rtt2[0], rtt3[1] * rtt2[0], rtt3[0] * rtt2[1], rtt3[1] * rtt2[1]
                    rtt = tt[0] - tt[3], tt[1] + tt[2]
                    # print("rel1_split:\t", rel1_split[0])
                    rule += weight_r * (torch.sum(torch.abs(rel1_split[0] - rtt[0]) ** 3) + torch.sum(
                        torch.abs(rel1_split[1] - rtt[1]) ** 3))
                    rule_num += 1

        for rel_1 in x[:, 1]:
            if rel_1 in self.rule2_p3:
                rel1_emb = self.embeddings[3](rel_1)
                rel1_split = rel1_emb[:self.rank], rel1_emb[self.rank:]
                for body in self.rule2_p3[rel_1]:
                    rel_2, rel_3 = body
                    weight_r = self.rule2_p3[rel_1][body]
                    rel2_emb = self.embeddings[3](torch.LongTensor([rel_2]).cuda())[0]
                    rel3_emb = self.embeddings[3](torch.LongTensor([rel_3]).cuda())[0]
                    rel2_split = rel2_emb[:self.rank], rel2_emb[self.rank:]
                    rtt3 = rel3_emb[:self.rank], rel3_emb[self.rank:]
                    tt2 = rel2_split[0] * transt[0][0], rel2_split[1] * transt[0][0], rel2_split[0] * transt[1][0], \
                          rel2_split[1] * transt[1][0]
                    rtt2 = tt2[0] - tt2[3], tt2[1] + tt2[2]
                    tt = rtt3[0] * rtt2[0], rtt3[1] * rtt2[0], rtt3[0] * rtt2[1], rtt3[1] * rtt2[1]
                    rtt = tt[0] - tt[3], tt[1] + tt[2]
                    # print("rel1_split:\t", rel1_split[0])
                    rule += weight_r * (torch.sum(torch.abs(rel1_split[0] - rtt[0]) ** 3) + torch.sum(
                        torch.abs(rel1_split[1] - rtt[1]) ** 3))
                    rule_num += 1

        for rel_1 in x[:, 1]:
            if rel_1 in self.rule2_p4:
                rel1_emb = self.embeddings[3](rel_1)
                rel1_split = rel1_emb[:self.rank], rel1_emb[self.rank:]
                for body in self.rule2_p4[rel_1]:
                    rel_2, rel_3 = body
                    weight_r = self.rule2_p4[rel_1][body]
                    rel2_emb = self.embeddings[3](torch.LongTensor([rel_2]).cuda())[0]
                    rel3_emb = self.embeddings[3](torch.LongTensor([rel_3]).cuda())[0]
                    rtt2 = rel2_emb[:self.rank], rel2_emb[self.rank:]
                    rtt3 = rel3_emb[:self.rank], rel3_emb[self.rank:]
                    tt = rtt3[0] * rtt2[0], rtt3[1] * rtt2[0], rtt3[0] * rtt2[1], rtt3[1] * rtt2[1]
                    rtt = tt[0] - tt[3], tt[1] + tt[2]
                    # print("rel1_split:\t", rel1_split[0])
                    rule += weight_r * (torch.sum(torch.abs(rel1_split[0] - rtt[0]) ** 3) + torch.sum(
                        torch.abs(rel1_split[1] - rtt[1]) ** 3))
                    rule_num += 1

        rule = rule / rule_num
        return (
            (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ right[0].t() +
            (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ right[1].t(),
            (h_static[0] * r_static[0] - h_static[1] * r_static[1]) @ right_static[0].t() +
            (h_static[1] * r_static[0] + h_static[0] * r_static[1]) @ right_static[1].t(),
            regularizer,
            self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight,
            rule
        )

    def forward_over_time(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2].weight

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        rel_no_time = self.embeddings[3](x[:, 1])
        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        score_time = (
                (lhs[0] * rel[0] * rhs[0] - lhs[1] * rel[1] * rhs[0] -
                 lhs[1] * rel[0] * rhs[1] + lhs[0] * rel[1] * rhs[1]) @ time[0].t() +
                (lhs[1] * rel[0] * rhs[0] - lhs[0] * rel[1] * rhs[0] +
                 lhs[0] * rel[0] * rhs[1] - lhs[1] * rel[1] * rhs[1]) @ time[1].t()
        )
        base = torch.sum(
            (lhs[0] * rnt[0] * rhs[0] - lhs[1] * rnt[1] * rhs[0] -
             lhs[1] * rnt[0] * rhs[1] + lhs[0] * rnt[1] * rhs[1]) +
            (lhs[1] * rnt[1] * rhs[0] - lhs[0] * rnt[0] * rhs[0] +
             lhs[0] * rnt[1] * rhs[1] - lhs[1] * rnt[0] * rhs[1]),
            dim=1, keepdim=True
        )
        return score_time + base

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_rhs_static(self, chunk_begin: int, chunk_size: int):
        return self.static_embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        rel_no_time = self.embeddings[3](queries[:, 1])
        time = self.embeddings[2](queries[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]
        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = (rt[0] - rt[3]) + rnt[0], (rt[1] + rt[2]) + rnt[1]

        h_static = self.static_embeddings[0](queries[:, 0])
        r_static = self.static_embeddings[1](queries[:, 1])

        h_static = h_static[:, :self.rank_static], h_static[:, self.rank_static:]
        r_static = r_static[:, :self.rank_static], r_static[:, self.rank_static:]

        return torch.cat([
            lhs[0] * full_rel[0] - lhs[1] * full_rel[1],
            lhs[1] * full_rel[0] + lhs[0] * full_rel[1]
        ], 1), torch.cat([
            h_static[0] * r_static[0] - h_static[1] * r_static[1],
            h_static[1] * r_static[0] + h_static[0] * r_static[1]
        ], 1)

#反向传播与损失计算部分核心代码
class TKBCOptimizer(object):
    def __init__(
            self, model: TKBCModel,
            emb_regularizer: Regularizer, temporal_regularizer: Regularizer, rule_regularizer: Regularizer,
            optimizer: optim.Optimizer, batch_size: int = 256,
            verbose: bool = True
    ):
        self.model = model
        self.emb_regularizer = emb_regularizer
        self.temporal_regularizer = temporal_regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose

        self.rule_regularizer = rule_regularizer

    def epoch(self, examples: torch.LongTensor):
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        loss = nn.CrossEntropyLoss(reduction='mean')
        loss_static = nn.CrossEntropyLoss(reduction='mean')
        with tqdm.tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[
                    b_begin:b_begin + self.batch_size
                ].cuda()
                predictions, pred_static, factors, time, rule = self.model.forward(input_batch)
                truth = input_batch[:, 2]

                l_fit = loss(predictions, truth)
                l_static = loss_static(pred_static, truth)
                l_reg = self.emb_regularizer.forward(factors)
                l_time = torch.zeros_like(l_reg)
                l_rule = self.rule_regularizer.forward(rule)
                if time is not None:
                    l_time = self.temporal_regularizer.forward(time)
                l = l_fit + 0.1 * l_static + l_reg + l_time

                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
                bar.set_postfix(
                    loss=f'{l_fit.item():.2f}',
                    loss_cs=f'{l_static.item():.2f}',
                    reg=f'{l_reg.item():.2f}',
                    cont=f'{l_time.item():.2f}',
                    rule=f'{l_rule:.2f}'
                )