import json
import os
from typing import List, Dict, Tuple, Set, Optional, Union
import re
import random
from collections import defaultdict
import argparse
import yaml
import shutil
from datetime import datetime

from tqdm import tqdm

from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

from log import get_logger

logger = get_logger('root', filename='schema.log')

from base import Event, Schema
from prompts import (build_expansion_prompts, build_naming_prompt,
                     build_relevance_prompt,
                     build_skeleton_prompt,
                     build_specificity_prompt,
                     build_chapter_binary_prompt,
                     build_chapter_multichoice_prompt,
                     build_grounding_prompt)

from graphtools import remove_bad_edges
from verify import verify_edges, get_verification_pairs, verify_edges_decomposition
from nli.inference import InferenceModel
from grounding import parse_grounding_response, read_xpo_dict
from gpttools import get_gpt3_response, load_cache, save_cache, setup


def get_events_from_response(response: str, min_desc_len: int = 10) -> List[Event]:
    event_list = []
    for step in response.strip().split('\n'):
        if step.strip() == '': continue
        m = re.match(r"-?\d+[\.\):]\s+([\w\s\d',]+)", step)
        if m:
            desc = m.group(1)
            if len(desc) < min_desc_len: continue
            new_evt = Event(description=desc)
            event_list.append(new_evt)
        else:
            m = re.match(r'-\s*([\w\s\d]+)', step)
            if m:
                desc = m.group(1)
                if len(desc) < min_desc_len: continue
                new_evt = Event(description=desc)
                event_list.append(new_evt)
            else:
                sents = sent_tokenize(step)
                for sent in sents:
                    desc = sent.strip()
                    if len(desc) < min_desc_len: continue
                    new_evt = Event(description=desc)
                    event_list.append(new_evt)

    return event_list


def get_name_from_response(response: str, max_length: int = 30) -> str:
    response = response.split('\n')[0]
    response = response.strip()

    m = re.match(r'Description:[\s\d\w\.]+(?:\'[\s\d\w\.]+)* (Name:)?', response)

    if m: return ''
    m = re.match(r'Name: ([\w\d\s]+)', response)
    if m: return m.group(1)

    if len(response) > max_length: return ''
    return response


def get_chapter_from_response(response: str, chapter_evts: List[Event]) -> Optional[Event]:
    response = response.strip('\n').strip()
    chapter_names = [(c, c.name) for c in chapter_evts]
    min_start = 256
    chosen = None
    for option in chapter_names:
        m = re.search(option[1], response)
        if m:
            start = m.start(0)
            if start < min_start:
                chosen = option[0]

    return chosen


def check_chapter(evt: Event, chapter_event: Event, all_chapters: List[Event], prompt_type='binary'):
    if prompt_type == 'multichoice':
        chapter_prompt = build_chapter_multichoice_prompt(evt, all_chapters)
        chosen_chapter = get_chapter_from_response(get_gpt3_response(chapter_prompt), all_chapters)
        return chosen_chapter == chapter_event
    else:
        chapter_prompt = build_chapter_binary_prompt(evt, chapter_event)
        res = get_gpt3_response(chapter_prompt)
        if res.strip('\n').strip().lower() == 'yes':
            return True
        else:
            logger.info(f"{chapter_event.name} -> event {evt.name} does not pass the chapter test")
            return False


def check_specificity(evt: Event) -> bool:
    specificity_prompt = build_specificity_prompt(evt)
    res = get_gpt3_response(specificity_prompt)
    res = res.split('\n')[0]
    if res.strip(' ').lower() == 'no':
        return True
    else:
        logger.info(f"event {evt.description} does not pass the specificity test")
        return False


def get_xpo_name_response(events: Dict[int, Event],
                          json_events, xpo_node_dict, overlay_parent_dict) -> Dict[int, Tuple]:
    gpt3_response = {}

    for idx, evt in tqdm(events.items()):
        response = get_gpt3_response(build_grounding_prompt(evt))
        event_name_candidate, result_dict_response = parse_grounding_response(response, json_events, xpo_node_dict,
                                                                              overlay_parent_dict)
        if len(event_name_candidate) == 0:
            max_iter = 10
            for _ in range(max_iter):
                response = get_gpt3_response(build_grounding_prompt(evt), use_cache=False)
                event_name_candidate, result_dict_response = parse_grounding_response(response, json_events,
                                                                                      xpo_node_dict,
                                                                                      overlay_parent_dict)
                if len(event_name_candidate) > 0:
                    break

        gpt3_response[idx] = (event_name_candidate, result_dict_response)

    return gpt3_response


def ground_events(schema: Schema,
                  xpo_path: str = "data/xpo_v5.1a.json",
                  model: str = "facebook/bart-large-mnli"):
    logger.info(f"reading xpo node dictionary from {xpo_path}")
    json_events, xpo_node_dict, overlay_parent_dict = read_xpo_dict(xpo_path)
    logger.info(f"getting grounding response from GPT-3")
    xpo_name_response = get_xpo_name_response(schema.events, json_events, xpo_node_dict, overlay_parent_dict)

    inferenceModel = InferenceModel(xpo_node_dict=xpo_node_dict,
                                    json_events=json_events,
                                    model_name=model)

    inference_model_result = inferenceModel.get_result(schema.events, xpo_name_response)

    return inference_model_result


def load_chapters(schema: Schema, chapter_file: str) -> List[Event]:
    with open(chapter_file) as f:
        chapters = json.load(f)
    logger.info('Loading Chapters ....')

    chapter2evt = {}
    chapter_events = []
    for c in chapters['chapters']:
        c_evt = Event(c['description'])
        c_evt.name = c['name']
        c_evt.is_chapter = True
        schema.add_child_event(schema, c_evt, check_dup=False)
        chapter2evt[c['idx']] = c_evt.id
        chapter_events.append(c_evt)

    for tuple in chapters['order']:
        before_evt = chapter2evt[tuple[0]]
        after_evt = chapter2evt[tuple[1]]
        schema.add_before_edge(schema.events[before_evt], schema.events[after_evt], weight=1.0)

    return chapter_events


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    default_ontology_path = "E:/inc schema2/schema/constant/xpo_v5.1a.json"

    parser.add_argument('--config_dir', type=str, default='configs')
    parser.add_argument('--load_configs', type=str, default='phase2b.yaml')

    parser.add_argument('--scenario_name', type=str, default='infrastructure disaster')
    parser.add_argument('--schema_dir', type=str, default='outputs/schema')
    parser.add_argument('--run_name', type=str, default='')
    parser.add_argument('--gpt_cache_dir', type=str, default='cache/')
    parser.add_argument('--use_cache', action='store_true', default=False)

    parser.add_argument('--ontology_path', type=str, default=default_ontology_path)


    parser.add_argument('--grounding_nli_model', type=str, default='facebook/bart-large-mnli')
    parser.add_argument('--require_relevance', action='store_true', default=False)
    parser.add_argument('--use_chapter', action='store_true', default=False)
    parser.add_argument('--chapter_dir', type=str, default='schema/data')
    parser.add_argument('--min_chapter', type=int, default=3)
    parser.add_argument('--max_chapter', type=int, default=10)
    parser.add_argument('--parent_threshold', type=float, default=0.7)
    parser.add_argument('--edge_threshold', type=float, default=0.2)
    parser.add_argument('--duration_threshold', type=float, default=0.6)
    parser.add_argument('--decompose', action='store_true', default=False)
    parser.add_argument('--calibrate', action='store_true', default=False)
    parser.add_argument('--skip_verification', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()

    random.seed(args.seed)

    if args.load_configs:
        with open(os.path.join(args.config_dir, args.load_configs), 'r') as f:
            yaml_configs = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in yaml_configs.items():
            args.__dict__[k] = v

    if not args.run_name:
        d = datetime.now()
        time_str = d.strftime('%m-%dT%H%M')
        schema_dir = f"{args.schema_dir}_{time_str}"
    else:
        schema_dir = f"{args.schema_dir}_{args.run_name}"
    os.makedirs(schema_dir, exist_ok=True)

    with open(os.path.join(schema_dir, 'params.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    logger.info('Setting up GPT3 access....')
    setup(os.path.join(args.config_dir, 'local.yaml'))

    if args.use_cache:
        load_cache(args.gpt_cache_dir, args.seed)

    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

    if args.scenario_name == 'all':
        scenario_list = [os.path.splitext(x)[0] for x in os.listdir(args.chapter_dir)]
    else:
        scenario_list = args.scenario_name.split(',')

    logger.info(f'inducing {len(scenario_list)} schemas...')

    for scenario in scenario_list:
        scenario = scenario.strip(' ').replace('_', ' ')
        scenario_path = scenario.replace(' ', '_')
        logger.info(f'The output will be saved to {schema_dir}/{scenario_path}')
        schema = Schema(scenario=scenario, embedding_model=sbert_model)

        if args.use_chapter:
            assert (args.chapter_dir != '')
            chapter_file = os.path.join(args.chapter_dir, f'{scenario_path}.json')
            all_chapters = load_chapters(schema, chapter_file)
        else:
            all_chapters = [schema, ]

        schema.visualize(schema_dir, suffix='chapters')
        schema.save(schema_dir)

        logger.info('Round 1 Skeleton Construction....')

        for chapter_evt in all_chapters:
            prompt = build_skeleton_prompt(scenario, chapter_evt) if args.use_chapter else build_skeleton_prompt(
                scenario)
            valid_events = 0
            first_attempt = True

            while valid_events < args.min_chapter:
                response = get_gpt3_response(prompt) if first_attempt else get_gpt3_response(prompt, use_cache=False)
                main_events = get_events_from_response(response)
                prev_e = None
                for e in main_events:
                    name_prompt = build_naming_prompt(e)
                    name = get_name_from_response(get_gpt3_response(name_prompt))
                    if name != '':
                        e.name = name
                        outcome = schema.add_after_event(prev_e, e) if prev_e else schema.add_event(e)
                        if outcome:
                            schema.add_child_event(chapter_evt, e)
                            prev_e = e
                            valid_events += 1
                first_attempt = False

        schema.visualize(schema_dir, suffix='skeleton')
        schema.save(schema_dir)

        logger.info('Round 2 Expansion...')
        expansion_queue = [e for e in main_events if schema.contains_event(e)]

        while expansion_queue:
            e = expansion_queue.pop(0)
            chapter_evt = e.get_chapter(schema.events)
            if len(chapter_evt.children) > args.max_chapter:
                continue

            e_prompts = build_expansion_prompts(e)

            for action, p in e_prompts.items():
                full_prompt = p
                response = get_gpt3_response(full_prompt)
                expanded_event = get_events_from_response(response)

                for e2 in expanded_event:
                    if args.require_relevance and not check_specificity(e2):
                        break
                    name_prompt = build_naming_prompt(e2)
                    name = get_name_from_response(get_gpt3_response(name_prompt))
                    if name == '':
                        break
                    e2.name = name
                    if check_chapter(e, chapter_evt, all_chapters):
                        outcome = schema.ACTIONS[action](e, e2)
                        if outcome and 'subevent' not in action:
                            schema.add_parent_edge(chapter_evt, e2, keep_chapter=False)

        schema.visualize(schema_dir, suffix='expansion')
        schema.save(schema_dir)
        scenario_path = schema.scenario.replace(' ', '_')
        shutil.copy('schema.log', os.path.join(schema_dir, scenario_path))

        if args.use_cache:
            save_cache(args.gpt_cache_dir, args.seed)

        if not args.skip_verification:
            logger.info('Round 3 Verification')
            verification_pairs = get_verification_pairs(schema, all_chapters, criteria='chapter')
            if args.decompose:
                schema = verify_edges_decomposition(schema, verification_pairs, schema_dir, args.calibrate,
                                                    args.edge_threshold, args.duration_threshold)
            else:
                schema = verify_edges(schema, verification_pairs, schema_dir, args.calibrate, args.edge_threshold)

            schema = remove_bad_edges(schema, all_chapters)
            schema.visualize(schema_dir, suffix='verify', by_chapter=True)
            if args.use_cache:
                save_cache(args.gpt_cache_dir, args.seed)

        ground_events(schema, args.ontology_path, args.grounding_nli_model)
        schema.visualize(schema_dir, suffix='grounded', by_chapter=True)
        schema.save(schema_dir)
