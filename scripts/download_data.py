#!/usr/bin/env python3
"""
ProG 标准数据下载脚本
=====================
自动下载 ProG 支持的所有数据集。数据将保存到 data/ 目录。
部分数据集通过 PyTorch Geometric / OGB 自动下载。

用法:
    python scripts/download_data.py --data_root data --datasets MUTAG   # 推荐：MUTAG 下载快
    python scripts/download_data.py [--data_root ./data] [--datasets all]
    python scripts/download_data.py --datasets Cora -v   # -v 失败时打印完整 traceback
    
    --datasets: all | node | graph | MUTAG,Cora,... (逗号分隔)
"""
import argparse
import os
import sys
import traceback
from tqdm import tqdm


def _err_msg(e):
    """保证异常信息可读，避免空 str(e)"""
    msg = str(e)
    if not msg:
        msg = repr(e) or type(e).__name__
    return msg

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

def download_node_datasets(data_root='./data', datasets_filter=None, verbose=False):
    """下载节点级数据集。datasets_filter=None 表示全部，否则为需下载的集合。"""
    from torch_geometric.datasets import Planetoid, Amazon, Reddit, WikiCS, Flickr, WebKB, Actor
    from torch_geometric.transforms import NormalizeFeatures
    
    root_planetoid = os.path.join(data_root, 'Planetoid')
    root_amazon = os.path.join(data_root, 'amazon')
    
    def need(name):
        return datasets_filter is None or name in datasets_filter
    
    print("=== 节点级数据集 ===")
    planetoid_names = [n for n in ['Cora', 'CiteSeer', 'PubMed'] if need(n)]
    if planetoid_names:
        for name in tqdm(planetoid_names, desc="Planetoid", unit="dataset"):
            try:
                d = Planetoid(root=root_planetoid, name=name, transform=NormalizeFeatures())
                print(f"  [OK] {name}: {d[0].num_nodes} nodes")
            except Exception as e:
                print(f"  [FAIL] {name}: {type(e).__name__}: {_err_msg(e)}")
                if verbose:
                    traceback.print_exc()
    
    amazon_names = [n for n in ['Computers', 'Photo'] if need(n)]
    if amazon_names:
        for name in tqdm(amazon_names, desc="Amazon", unit="dataset"):
            try:
                d = Amazon(root=root_amazon, name=name)
                print(f"  [OK] {name}")
            except Exception as e:
                print(f"  [FAIL] {name}: {type(e).__name__}: {_err_msg(e)}")
                if verbose:
                    traceback.print_exc()
    
    webkb_names = [n for n in ['Wisconsin', 'Texas'] if need(n)]
    if webkb_names:
        for name in tqdm(webkb_names, desc="WebKB", unit="dataset"):
            try:
                d = WebKB(root=os.path.join(data_root, name), name=name)
                print(f"  [OK] {name}")
            except Exception as e:
                print(f"  [FAIL] {name}: {type(e).__name__}: {_err_msg(e)}")
                if verbose:
                    traceback.print_exc()
    
    other_pyg = [
        ('Reddit', lambda: Reddit(root=os.path.join(data_root, 'Reddit'))),
        ('WikiCS', lambda: WikiCS(root=os.path.join(data_root, 'WikiCS'))),
        ('Flickr', lambda: Flickr(root=os.path.join(data_root, 'Flickr'))),
        ('Actor', lambda: Actor(root=os.path.join(data_root, 'Actor'))),
    ]
    other_pyg_filtered = [(n, fn) for n, fn in other_pyg if need(n)]
    if other_pyg_filtered:
        for name, loader_fn in tqdm(other_pyg_filtered, desc="其他节点数据集(PyG)", unit="dataset"):
            try:
                d = loader_fn()
                print(f"  [OK] {name}")
            except Exception as e:
                print(f"  [FAIL] {name}: {type(e).__name__}: {_err_msg(e)}")
                if verbose:
                    traceback.print_exc()

    if need('ogbn-arxiv'):
        try:
            from ogb.nodeproppred import PygNodePropPredDataset
            d = PygNodePropPredDataset(name='ogbn-arxiv', root=data_root)
            print(f"  [OK] ogbn-arxiv")
        except Exception as e:
            print(f"  [FAIL] ogbn-arxiv: {type(e).__name__}: {_err_msg(e)}")
            if verbose:
                traceback.print_exc()


def download_graph_datasets(data_root='./data', datasets_filter=None, verbose=False):
    """下载图级数据集。datasets_filter=None 表示全部。"""
    from torch_geometric.datasets import TUDataset
    
    def need(name):
        return datasets_filter is None or name in datasets_filter
    
    tud_names = ['MUTAG', 'ENZYMES', 'COLLAB', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY', 
                 'COX2', 'BZR', 'PTC_MR', 'DD']
    tud_filtered = [n for n in tud_names if need(n)]
    if tud_filtered:
        print("=== 图级数据集 (TUDataset) ===")
        tud_root = os.path.join(data_root, 'TUDataset')
        for name in tqdm(tud_filtered, desc="TUDataset", unit="dataset"):
            try:
                d = TUDataset(root=tud_root, name=name, use_node_attr=True)
                print(f"  [OK] {name}: {len(d)} graphs")
            except Exception as e:
                print(f"  [FAIL] {name}: {type(e).__name__}: {_err_msg(e)}")
                if verbose:
                    traceback.print_exc()
    
    ogb_names = ['ogbg-ppa', 'ogbg-molhiv', 'ogbg-molpcba', 'ogbg-code2']
    ogb_filtered = [n for n in ogb_names if need(n)]
    if ogb_filtered:
        print("=== 图级数据集 (OGB) ===")
        try:
            from ogb.graphproppred import PygGraphPropPredDataset
        except ImportError as e:
            print(f"  [FAIL] OGB import: {e}")
        else:
            ogb_root = os.path.join(data_root, 'ogbg')
            for name in tqdm(ogb_filtered, desc="OGB", unit="dataset"):
                try:
                    d = PygGraphPropPredDataset(name=name, root=ogb_root)
                    print(f"  [OK] {name}: {len(d)} graphs")
                except Exception as e:
                    print(f"  [FAIL] {name}: {type(e).__name__}: {_err_msg(e)}")
                    if verbose:
                        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='ProG 数据下载脚本')
    parser.add_argument('--data_root', type=str, default='data',
                        help='数据保存根目录，需与 load4data 中路径一致 (default: data)')
    parser.add_argument('--datasets', type=str, default='all',
                        help='all | node | graph | 或逗号分隔的数据集名称')
    parser.add_argument('--verbose', '-v', action='store_true', help='失败时打印完整 traceback')
    args = parser.parse_args()
    
    os.makedirs(args.data_root, exist_ok=True)
    
    if args.datasets == 'all':
        download_node_datasets(args.data_root, datasets_filter=None, verbose=args.verbose)
        download_graph_datasets(args.data_root, datasets_filter=None, verbose=args.verbose)
    elif args.datasets == 'node':
        download_node_datasets(args.data_root, datasets_filter=None, verbose=args.verbose)
    elif args.datasets == 'graph':
        download_graph_datasets(args.data_root, datasets_filter=None, verbose=args.verbose)
    else:
        # Specific datasets - download both node and graph if needed
        names = [s.strip() for s in args.datasets.split(',')]
        node_names = ['PubMed', 'CiteSeer', 'Cora', 'Computers', 'Photo', 'Reddit', 
                      'WikiCS', 'Flickr', 'Wisconsin', 'Texas', 'Actor', 'ogbn-arxiv']
        graph_names = ['MUTAG', 'ENZYMES', 'COLLAB', 'PROTEINS', 'IMDB-BINARY', 
                      'REDDIT-BINARY', 'COX2', 'BZR', 'PTC_MR', 'DD',
                      'ogbg-ppa', 'ogbg-molhiv', 'ogbg-molpcba', 'ogbg-code2']
        
        node_req = set(n for n in names if n in node_names)
        graph_req = set(n for n in names if n in graph_names)
        if node_req:
            download_node_datasets(args.data_root, datasets_filter=node_req, verbose=args.verbose)
        if graph_req:
            download_graph_datasets(args.data_root, datasets_filter=graph_req, verbose=args.verbose)
    
    print("\n数据下载完成。详见 README.md 中的数据集链接。")


if __name__ == '__main__':
    main()
