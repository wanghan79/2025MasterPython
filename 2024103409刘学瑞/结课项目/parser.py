import logging
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from MySQLdb.constants import FIELD_TYPE
from MySQLdb.converters import conversions as mysql_conv
from sqlalchemy import create_engine, text, bindparam
from tqdm import tqdm
from utils.ConfigParser import ConfigParser

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


class ChemblParser:
    def __init__(self, config):
        """Initialize data processing pipeline.

        Args:
            config_path (str, optional): Path to configuration file. Defaults to 'data.config'.
        """
        self.compound_batches = 0
        self.target_batches = 0
        self.total_batches = 0
        self.config = config

        self.mysql_cfg = self._parse_mysql_config()
        self.batch_size = int(self.config.get('batch_size'))
        self.thread_pool_size = int(self.config.get('thread_pool_size'))

        self.engine = None
        self._init_mysql()
        self._connect_mysql()

    QUERY_MAP = {
        'compound': {
            'properties': """
                            SELECT molregno, mw_freebase, alogp, hba, hbd, psa, rtb, 
                                    ro3_pass, num_ro5_violations, molecular_species,
                                    aromatic_rings, heavy_atoms, qed_weighted, np_likeness_score
                            FROM compound_properties 
                            WHERE molregno IN :ids
                        """,
            'activities': """
                            SELECT 
                                a.molregno, a.assay_id, a.standard_value, a.standard_units,
                                a.standard_relation, a.pchembl_value, t.chembl_id as target_id,
                                t.pref_name as target_name, ass.chembl_id as assay_chembl_id,
                                ass.description as assay_description
                            FROM activities a
                            LEFT JOIN assays ass ON a.assay_id = ass.assay_id
                            LEFT JOIN target_dictionary t ON ass.tid = t.tid
                            WHERE a.molregno IN :ids
                            AND a.standard_value IS NOT NULL
                        """,
            'structures': """
                            SELECT molregno, standard_inchi, standard_inchi_key, canonical_smiles, molfile
                            FROM compound_structures 
                            WHERE molregno IN :ids
                        """,
            'synonyms': """
                            SELECT molregno, synonyms, syn_type
                            FROM molecule_synonyms
                            WHERE molregno IN :ids
                        """,
            'mechanisms': """
                            SELECT 
                                m.molregno, m.mechanism_of_action, m.action_type,
                                m.direct_interaction, m.molecular_mechanism,
                                t.chembl_id as target_id, t.pref_name as target_name
                            FROM drug_mechanism m
                            LEFT JOIN target_dictionary t ON m.tid = t.tid
                            WHERE m.molregno IN :ids
                        """,
        },
        'target': {
            'basic': """
                    SELECT td.tid, td.chembl_id, td.pref_name, td.target_type, td.tax_id, 
                           td.organism, td.species_group_flag
                    FROM target_dictionary td 
                    WHERE td.tid IN :ids
                """,
            'components': """
                    SELECT tc.tid, tc.component_id, tc.homologue, cs.accession, cs.component_type, 
                           cs.sequence, cs.organism as component_organism, cs.tax_id as component_tax_id 
                    FROM target_components tc 
                    LEFT JOIN component_sequences cs ON tc.component_id = cs.component_id
                    WHERE tc.tid IN :ids
                """,
            'relationships': """
                    SELECT tr.tid, tr.relationship, tr.related_tid, 
                           td.chembl_id as related_chembl_id, td.pref_name as related_name 
                    FROM target_relations tr 
                    LEFT JOIN target_dictionary td ON tr.related_tid = td.tid
                    WHERE tr.tid IN :ids
                """,
            'tissues': """
                    SELECT a.tid, td.tissue_id, td.uberon_id, td.pref_name, td.efo_id 
                    FROM assays a 
                    LEFT JOIN tissue_dictionary td ON a.tissue_id = td.tissue_id 
                    WHERE a.tid IN :ids
                """,
            'compounds': """
                    SELECT ass.tid, md.chembl_id as compound_id, md.pref_name as compound_name, 
                           a.standard_type, a.standard_value, a.standard_units, a.pchembl_value, 
                           a.standard_relation, doc.pubmed_id 
                    FROM assays ass
                    LEFT JOIN activities a ON a.assay_id = ass.assay_id
                    LEFT JOIN compound_records cr ON a.record_id = cr.record_id 
                    LEFT JOIN molecule_dictionary md ON cr.molregno = md.molregno 
                    LEFT JOIN docs doc ON a.doc_id = doc.doc_id 
                    WHERE ass.tid IN :ids 
                    AND a.standard_value IS NOT NULL
                """
        }
    }

    def _parse_mysql_config(self):
        """Parse MySQL connection parameters from config file.

        Returns:
            dict: Dictionary containing MySQL connection parameters
                 (host, port, user, password, database)
        """
        return {
            'host': self.config.get('host', 'mysql'),
            'port': int(self.config.get('port', 'mysql')),
            'user': self.config.get('user', 'mysql'),
            'password': self.config.get('password', 'mysql'),
            'database': self.config.get('mysql_db_name')
        }

    def _connect_mysql(self, use_db=True):
        """Establish connection to MySQL database."""
        conv = mysql_conv.copy()
        conv[FIELD_TYPE.DECIMAL] = float
        conv[FIELD_TYPE.NEWDECIMAL] = float

        try:
            conn_str = f"mysql+mysqldb://{self.mysql_cfg['user']}:{self.mysql_cfg['password']}@{self.mysql_cfg['host']}:{self.mysql_cfg['port']}" + (
                f"/{self.mysql_cfg['database']}" if use_db else "/") + "?charset=utf8"
            self.engine = create_engine(conn_str, connect_args={'conv': conv})
        except Exception as e:
            logging.error(f"Failed to connect to MySQL: {e}")
            raise

    def _init_mysql(self):
        """Initialize MySQL database and import data using native MySQL tool."""
        try:
            self._connect_mysql(use_db=False)
            with self.engine.connect() as connection:
                check_query = text(
                    "SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME = :dbname"
                )
                result = connection.execute(check_query, {"dbname": self.config.get("mysql_db_name")}).fetchone()

                if result:
                    logging.info(
                        f"Database {self.config.get('mysql_db_name')} already exists. Skipping initialization.")
                    return  # 直接返回，跳过后续初始化流程

                connection.execute(
                    f'CREATE DATABASE IF NOT EXISTS {self.config.get("mysql_db_name")} DEFAULT CHARSET=utf8')

            chembl_name = self.config.get("source_url_1").split("/")[-1].split(".")[0]
            mysql_dmp = Path(self.config.get("data_path_1")) / f"{chembl_name}.dmp"

            logging.info("MySQL data loading starts.  It may take 90 mins or even more.")
            pwd = f"-p{self.mysql_cfg['password']}" if self.mysql_cfg['password'] else ''
            cmd = f"mysql -h {self.mysql_cfg['host']} -P {self.mysql_cfg['port']} -u {self.mysql_cfg['user']} {pwd} {self.config.get('mysql_db_name')} < {mysql_dmp.as_posix()}"

            result = subprocess.run(
                cmd,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            if result.returncode != 0:
                error_msg = result.stderr.decode('utf-8')
                raise RuntimeError(f"MySQL import failed: {error_msg}")

            logging.info("MySQL data loading finishes.")

        except subprocess.CalledProcessError as e:
            logging.error(f"MySQL command failed: {e.stderr.decode('utf-8')}")
            raise
        except Exception as e:
            logging.error(f"Error during MySQL initialization: {str(e)}")
            raise
        finally:
            if hasattr(self, 'engine'):
                self.engine.dispose()

    def _fetch(self, query, params):
        """Execute SQL query and return results.

        Args:
            query (str): SQL query string
            params (dict): Query parameters

        Returns:
            list[dict]: List of dictionaries representing query results
        """
        try:
            with self.engine.connect() as conn:
                stmt = text(query)
                if "IN :ids" in query:
                    stmt = stmt.bindparams(bindparam("ids", expanding=True))
                return [dict(row) for row in conn.execute(stmt, params)]
        except Exception as e:
            logging.error(f"Failed to execute query: {e}")
            return []

    def _get_total_batches(self, mode):
        """Calculate total number of batches needed."""
        if mode == 'target':
            query = "SELECT COUNT(DISTINCT tid) AS total FROM target_dictionary"
        elif mode == 'compound':
            query = "SELECT COUNT(*) as total FROM molecule_dictionary"
        else:
            raise ValueError("Invalid mode")

        result = self._fetch(query, {})
        total = result[0]['total'] if result else 0
        return (total + self.batch_size - 1) // self.batch_size

    def _group_by_key(self, data, key, is_list=True):
        """通用分组方法."""
        grouped = defaultdict(list) if is_list else {}
        for item in data:
            grouped_key = item.get(key)
            if is_list:
                grouped[grouped_key].append(item)
            else:
                if grouped_key not in grouped:
                    grouped[grouped_key] = item
        return grouped

    def _process_basic_targets(self, data):
        """处理基础靶点信息并合并基因数据."""
        targets_map = {}
        for row in data:
            tid = row['tid']
            if tid not in targets_map:
                targets_map[tid] = {
                    'chembl_id': row['chembl_id'],
                    'pref_name': row['pref_name'],
                    'target_type': row['target_type'],
                    'organism': {
                        'tax_id': row['tax_id'],
                        'scientific_name': row['organism'],
                        'common_name': None
                    },
                    'species_group_flag': bool(row['species_group_flag'])
                }
        return targets_map

    def get_batch_num(self):
        self.compound_batches = self._get_total_batches('compound')
        self.target_batches = self._get_total_batches('target')
        self.total_batches = self.compound_batches + self.target_batches
        return self.total_batches

    def start(self):
        """Execute main data processing pipeline."""
        if not self.total_batches:
            self.get_batch_num()
        # parser compound
        for batch in range(self.compound_batches):
            rows = self._fetch("""
                    SELECT molregno, pref_name, chembl_id
                    FROM molecule_dictionary
                    LIMIT :limit OFFSET :offset
                """, {
                'limit': self.batch_size, 'offset': batch * self.batch_size
            })
            if not rows:
                break

            molregnos = [r['molregno'] for r in rows]
            with ThreadPoolExecutor(max_workers=self.thread_pool_size) as pool:
                futures = [
                    pool.submit(lambda data_name, ids: self._fetch(self.QUERY_MAP['compound'].get(data_name),
                                                                   {"ids": tuple(ids)}), key, molregnos)
                    for key in self.QUERY_MAP['compound']
                ]
                properties, activities, structures, synonyms, mechanisms = tuple(future.result() for future in futures)

            properties_grouped = self._group_by_key(properties, 'molregno', is_list=False)
            activities_grouped = self._group_by_key(activities, 'molregno')
            structures_grouped = self._group_by_key(structures, 'molregno', is_list=False)
            synonyms_grouped = self._group_by_key(synonyms, 'molregno')
            mechanisms_grouped = self._group_by_key(mechanisms, 'molregno')

            yield [{
                'pref_name': r["pref_name"],
                'chembl_id': r["chembl_id"],
                'structures': structures_grouped.get(r['molregno']),
                'synonyms': synonyms_grouped.get(r['molregno'], []),
                'properties': properties_grouped.get(r['molregno']),
                'activities': activities_grouped.get(r['molregno'], []),
                'mechanisms': mechanisms_grouped.get(r['molregno'], []),
            } for r in rows]

        # parser target
        logging.info("Start parsing target data...")
        for batch in range(self.target_batches):
            tids = [row['tid'] for row in self._fetch(
                "SELECT DISTINCT tid FROM target_dictionary LIMIT :limit OFFSET :offset",
                {'limit': self.batch_size, 'offset': batch * self.batch_size}
            )]
            if not tids:
                break

            with ThreadPoolExecutor(max_workers=self.thread_pool_size) as executor:
                futures = {
                    key: executor.submit(
                        lambda data_name, ids: self._fetch(self.QUERY_MAP['target'].get(data_name),
                                                           {"ids": tuple(ids)}),
                        key, tids)
                    for key in self.QUERY_MAP['target']
                }
                results = {key: future.result() for key, future in futures.items()}

            targets_map = self._process_basic_targets(results['basic'])
            component_groups = self._group_by_key(results['components'], 'tid')
            relation_groups = self._group_by_key(results['relationships'], 'tid')
            tissue_groups = self._group_by_key(results['tissues'], 'tid')
            # compound_groups = self._group_by_key(results['compounds'], 'tid')

            for tid, target in targets_map.items():
                target.update({
                    'components': [{
                        'component_id': f"CS{c['component_id']}",
                        'accession': c['accession'],
                        'component_type': c['component_type'],
                        'sequence': c['sequence'],
                        'homologue': bool(c['homologue']),
                        'organism': {
                            'scientific_name': c['component_organism'],
                            'tax_id': c['component_tax_id']
                        }
                    } for c in component_groups.get(tid, [])],
                    'relationships': [{
                        'related_target': f"T{r['related_tid']}",
                        'chembl_id': r['related_chembl_id'],
                        'relationship_type': r['relationship'].upper(),
                        'description': f"{r['relationship']} {r['related_name']}"
                    } for r in relation_groups.get(tid, [])],
                    'tissue_distribution': [{
                        'tissue_id': t['tissue_id'],
                        'name': t['pref_name'],
                        'uberon_id': t['uberon_id'],
                        'efo_id': t['efo_id'],
                        'evidence': 'Assay data'
                    } for t in tissue_groups.get(tid, [])],
                })
                # too many related compounds for mongodb to store (up to 1.8M records)
                # target.update({'related_compounds': [{
                #         'compound_id': c['compound_id'],
                #         'name': c['compound_name'],
                #         'activity_type': c['standard_type'],
                #         'activity_value': c['standard_value'],
                #         'activity_units': c['standard_units'],
                #         'pchembl_value': c['pchembl_value'],
                #         'reference': f"PMID:{c['pubmed_id']}" if c['pubmed_id'] else None
                #     } for c in compound_groups.get(tid, [])]})

            yield list(targets_map.values())


if __name__ == '__main__':
    cfg ="conf/drugkb.config"
    config = ConfigParser(cfg)
    config.set_section("chembl")
    chembl_parser = ChemblParser(config)
    for i in tqdm(chembl_parser.start(), total=chembl_parser.get_batch_num()):
        pass