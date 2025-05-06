import torch
from sklearn.model_selection import train_test_split
import numpy as np
import os
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_undirected
import logging
import time
from pathlib import Path
from utils import (
    get_pos_neg_edges, add_tar_edges_to_g, get_balanced_edges, 
    collated_data_separate)
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from torch_geometric.utils import subgraph

class SealSramDataset(InMemoryDataset):
    def __init__(
        self,
        name, #add
        root, #add
        neg_edge_ratio=1.0,
        to_undirected=True,
        sample_rates=[1.0], 
        task_level='edge',
        net_only=True,
        transform=None, 
        pre_transform=None,
        class_boundaries=[0.2, 0.4, 0.6, 0.8]
    ) -> None:
        """ The SRAM dataset. 
        It can be a combination of several large circuit graphs or millions of sampled subgraphs.
        Args:
            name (str): The name of the dataset.
            root (str): The root directory of the dataset.
            neg_edge_ratio (float): The ratio of negative edges to positive edges.
            to_undirected (bool): Whether to convert the graph to an undirected graph.
            sample_rates (list): The sampling rates of target edges for each dataset.
            task_type (str): The task type. It can be 'classification', 'regression', 'node_classification', 'node_regression'.
            num_classes (int): The number of classes.
            class_boundaries (list): The boundaries of the classes.
        """
        self.name = 'sram'
        self.class_boundaries = torch.tensor(class_boundaries)
        print("self.class_boundaries", self.class_boundaries)
        ## split the dataset according to '+' in the name
        if '+' in name:
            self.names = name.split('+')
        else:
            self.names = [name]
            
        print(f"SealSramDataset includes {self.names} circuits")

        self.sample_rates = sample_rates
        assert len(self.names) == len(self.sample_rates), \
            f"len of dataset:{len(self.names)}, len of sample_rate: {len(self.sample_rates)}"
        
        self.folder = os.path.join(root, self.name)
        self.neg_edge_ratio = neg_edge_ratio
        self.to_undirected = to_undirected
        ## data_lengths can be the number of total datasets or the number of subgraphs
        self.data_lengths = {}
        ## offset index for each graph
        self.data_offsets = {}

        self.task_level = task_level
        self.net_only = net_only
    
        self.max_net_node_feat = torch.ones((1, 17)) # the max feature dimension of net and dev nodes
        self.max_dev_node_feat = torch.ones((1, 17))

        super().__init__(self.folder, transform, pre_transform)
        data_list = []

        for i, name in enumerate(self.names):
            ## If a processed data file exsit, we load it directly
            loaded_data, loaded_slices = torch.load(self.processed_paths[i])
            # print("loaded_data", loaded_data)
            # print("loaded_slices", loaded_slices)

            self.data_offsets[name] = len(data_list)
            if self.task_level == 'node':
                self.data_lengths[name] = loaded_data.y.size(0)
            elif self.task_level == 'edge':
                self.data_lengths[name] = loaded_data.edge_label.size(0)
            else:
                raise ValueError(f"Invalid task level: {self.task_level}")

            if loaded_slices is not None:
                data_list += collated_data_separate(loaded_data, loaded_slices)#[:data_len]
            else:
                data_list.append(loaded_data)
            
            print(f"load processed {name}, "+
                  f"len(data_list)={self.data_lengths[name]}, "+
                  f"data_offset={self.data_offsets[name]} ")
    
        ## combine multiple graphs into data list
        self.data, self.slices = self.collate(data_list)

    def norm_nfeat(self, ntypes):
        """
         Only `DEVICE` and `NET` nodes have circuit statistics.
        Args:
            ntypes (list): The node types {0, 1} to be normalized
        """
        if self._data is None or self.slices is None:
            self.data, self.slices = self.collate(self._data_list)
            self._data_list = None


        # normalize the node features
        for ntype in ntypes:
            node_mask = self._data.node_type == ntype
            max_node_feat, _ = self._data.node_attr[node_mask].max(dim=0, keepdim=True)
            max_node_feat[max_node_feat == 0.0] = 1.0

            print(f"normalizing node_attr {ntype}: {max_node_feat} ...")
            self._data.node_attr[node_mask] /= max_node_feat

        if self.task_level == 'edge':
            ## normalize edge_label i.e., coupling capacitance
            self._data.edge_label = torch.log10(self._data.edge_label * 1e21) 
            self._data.edge_label /= 6
            self._data.edge_label[self._data.edge_label < 0] = 0.0
            self._data.edge_label[self._data.edge_label > 1] = 1.0
            edge_label_c = torch.bucketize(self._data.edge_label, self.class_boundaries)
            self._data.edge_label = torch.stack(
                [self._data.edge_label, edge_label_c], dim=1
            )
            self._data_list = None
          


        elif self.task_level == 'node':
            ## normalize the node label i.e., lumped ground capacitance
            self._data.y = torch.log10(self._data.y * 1e20) / 6
            self._data.y[self._data.y < 0] = 0.0
            self._data.y[self._data.y > 1] = 1.0
            node_label_c = torch.bucketize(self._data.y, self.class_boundaries)
            self._data.y = torch.stack(
                [self._data.y, node_label_c], dim=1
            )
            print("self._data.y", self._data.y)
            self._data_list = None

       
           

    def set_cl_embeds(self, embeds):
        """
        Set dataset attribute `x` as the embeddings learned by SGRL.
        Args:
            embeds (torch.Tensor): The embeddings [N, cl_hid_dim] learned by SGRL.
        """
        if self._data is None or self.slices is None:
            self.data, self.slices = self.collate(self._data_list)
            self._data_list = None

        self._data.x = embeds
        self._data_list = None
        print("Setting CL embeddings to x...")
        print('self._data', self._data)
        # print('self.slices', self.slices)

    def sram_graph_load(self, name, raw_path):
        """
        In the loaded circuit graph, ground capacitance values are stored in "tar_node_y' attribute of the node. 
        There are two edge sets. 
        The first is the connections existing in circuit topology, attribute names start with "edge". 
        For example, "g.edge_index" and "g.edge_type".
        The second is edges to be predicted, which are parasitic coupling edges. Their attribute names start with "tar_edge". 
        For example, "g.tar_edge_index" and "g.tar_edge_type".
        Coupling capacitance values are stored in the 'tar_edge_y' attribute of the edge.
        Args:
            name (str): The name of the dataset.
            raw_path (str): The path of the raw data file.
        Returns:
            g (torch_geometric.data.Data): The processed homo graph data.
        """
        logging.info(f"raw_path: {raw_path}")
        hg = torch.load(raw_path)
        if isinstance(hg, list):
            hg = hg[0]
        # print("hg", hg)
        power_net_ids = torch.tensor([0, 1])
        
        if name == "sandwich":
            # VDD VSS TTVDD
            power_net_ids = torch.tensor([0, 1, 1422])
        elif name == "ultra8t":
            # VDD VSS SRMVDD
            power_net_ids = torch.tensor([0, 1, 377])
        elif name == "sram_sp_8192w":
            # VSSE VDDCE VDDPE
            power_net_ids = torch.tensor([0, 1, 2])
        elif name == "ssram":
            # VDD VSS VVDD
            power_net_ids = torch.tensor([0, 1, 352])
        elif name == "digtime":
            power_net_ids = torch.tensor([0, 1])
        elif name == "timing_ctrl":
            power_net_ids = torch.tensor([0, 1])
        elif name == "array_128_32_8t":
            power_net_ids = torch.tensor([0, 1])
        
        """ graph transform """ 
        ### remove the power pins
        subset_dict = {}
        for ntype in hg.node_types:
            subset_dict[ntype] = torch.ones(hg[ntype].num_nodes, dtype=torch.bool)
            if ntype == 'net':
                subset_dict[ntype][power_net_ids] = False

        hg = hg.subgraph(subset_dict)
        hg = hg.edge_type_subgraph([
            ## circuit connections in schematics
            ('device', 'device-pin', 'pin'),
            ('pin', 'pin-net', 'net'),
            ## parasitic coupling edges: pin2net, pin2pin, net2net
            ('pin', 'cc_p2n', 'net'),
            ('pin', 'cc_p2p', 'pin'),
            ('net', 'cc_n2n', 'net'),
        ])

        print(hg)

        ### transform hetero g into homo g
        g = hg.to_homogeneous()
        g.name = name
        assert hasattr(g, 'node_type')
        assert hasattr(g, 'edge_type')
        edge_offset = 0
        tar_edge_y = []
        tar_node_y = []
        g._n2type = {}
        node_feat = []
        max_feat_dim = 17 # the max feature dimension of nodes

        hg['device'].y = torch.ones((hg['device'].x.shape[0], 1)) * 1e-30   # 1e-30 is the minimum ground capacitance value

        for n, ntype in enumerate(hg.node_types):
            g._n2type[ntype] = n
            feat = hg[ntype].x
            feat = torch.nn.functional.pad(feat, (0, max_feat_dim-feat.size(1)))
            node_feat.append(feat)
            tar_node_y.append(hg[ntype].y)
        
        ## There is 'node_type' attribute after transforming hg to g.
        ## The 'node_type' is used as default node feature, g.x.
        g.x = g.node_type.view(-1, 1)
        ## circuit statistic features
        g.node_attr = torch.cat(node_feat, dim=0)

        if self.task_level == 'node' :
            if self.net_only:
                net_mask = g.node_type == g._n2type['net'] 
                g.tar_node_y = torch.zeros((g.num_nodes, 1)) 
                net_nodes = torch.where(net_mask)[0]
                g.tar_node_y[net_nodes] = hg['net'].y  # assign the ground capacitance value to the net nodes
            else:
                ## lumped ground capacitance on net/pin nodes
                g.tar_node_y = torch.cat(tar_node_y, dim=0)
            g.y = g.tar_node_y
        

       
        g._e2type = {}

        for e, (edge, store) in enumerate(hg.edge_items()):
            g._e2type[edge] = e
            if 'cc' in edge[1]:
                tar_edge_y.append(store['y'])
            else:
                # edge_index's shape [2, num_edges]
                edge_offset += store['edge_index'].shape[1]
        
        g._num_ntypes = len(g._n2type)
        g._num_etypes = len(g._e2type)
        logging.info(f"g._n2type {g._n2type}")
        logging.info(f"g._e2type {g._e2type}")

        tar_edge_index = g.edge_index[:, edge_offset:]
        tar_edge_type = g.edge_type[edge_offset:]
        tar_edge_y = torch.cat(tar_edge_y)

        # testing
        # for i in range(tar_edge_type.min(), tar_edge_type.max()+1):
        #     mask = tar_edge_type == i
        #     print("tar_edge_type", tar_edge_type[mask][0], "tar_edge_y", tar_edge_y[mask][0])
        # assert 0

        ## restrict the capcitance value range 
        legel_edge_mask = (tar_edge_y < 1e-15) & (tar_edge_y > 1e-21)
        # tar_edge_src_y = g.tar_node_y[tar_edge_index[0, :]].squeeze()
        # tar_edge_dst_y = g.tar_node_y[tar_edge_index[1, :]].squeeze()
        # legel_node_mask = (tar_edge_src_y < 1e-13) & (tar_edge_src_y > 1e-23)
        # legel_node_mask &= (tar_edge_dst_y < 1e-13) & (tar_edge_dst_y > 1e-23)

        ## remove the target edges with extreme capacitance values
        g.tar_edge_y = tar_edge_y[legel_edge_mask]# & legel_node_mask]
        g.tar_edge_index = tar_edge_index[:, legel_edge_mask]# & legel_node_mask]
        g.tar_edge_type = tar_edge_type[legel_edge_mask]# & legel_node_mask]
        # logging.info(f"we filter out the edges with Cc > 1e-15 and Cc < 1e-21 " + 
        #              f"{legel_edge_mask.size(0)-legel_edge_mask.sum()}")
        # logging.info(f"we filter out the edges with src/dst Cg > 1e-13 and Cg < 1e-23 " +
        #              f"{legel_node_mask.size(0)-legel_node_mask.sum()}")

        ## Calculate target edge type distributions (Cc_p2n : Cc_p2p : Cc_n2n)
        _, g.tar_edge_dist = g.tar_edge_type.unique(return_counts=True)
        
        ## remove target edges from the original g
        g.edge_type = g.edge_type[0:edge_offset]
        g.edge_index = g.edge_index[:, 0:edge_offset]
        

        ## convert to undirected edges
        if self.to_undirected:
                g.edge_index, g.edge_type = to_undirected(
                    g.edge_index, g.edge_type, g.num_nodes, reduce='mean'
                )
        
        return g

    def single_g_process(self, idx: int):
        print(f"processing dataset {self.names[idx]} "+ 
                     f"with sample_rate {self.sample_rates[idx]}...")
        ## we can load multiple graphs
        graph = self.sram_graph_load(self.names[idx], self.raw_paths[idx])
        print(f"loaded graph {graph}")
        
        ## generate negative edges for the loaded graph
        neg_edge_index, neg_edge_type = get_pos_neg_edges(
            graph, neg_ratio=self.neg_edge_ratio)
        
        
        ## sample a portion of pos/neg edges
        (
            pos_edge_index, pos_edge_type, pos_edge_y,
            neg_edge_index, neg_edge_type
        ) = get_balanced_edges(
            graph, neg_edge_index, neg_edge_type, 
            self.neg_edge_ratio, self.sample_rates[idx]
        )


        if self.task_level == 'edge' :
            ## We only consider the positive edges in the regression task.
            links = pos_edge_index  # [2, Np]
            labels = pos_edge_y
        elif self.task_level == 'node':
            # node classification
            graph.y = graph.tar_node_y.squeeze()  # assume shape is [num_nodes]
        else:
            raise ValueError(f"No defination of task {self.task_level} in this version!")
        
        ## remove the redundant attributes in this version
        del graph.tar_node_y
        del graph.tar_edge_index
        del graph.tar_edge_type
        del graph.tar_edge_y

        
        if self.task_level == 'node':
            torch.save((graph, None), self.processed_paths[idx])
            return graph.y.size(0)
        elif self.task_level == 'edge':
            ## To use LinkNeighborLoader, the target links rename to edge_label_index
            ## target edge labels rename to edge_label
            graph.edge_label_index = links
            graph.edge_label = labels
            torch.save((graph, None), self.processed_paths[idx])
            return graph.edge_label.size(0)
        else:
            raise ValueError(f"No defination of task {self.task_type} in this version!")

    def process(self):
        data_lens_for_split = []
        p = Path(self.processed_dir)
        ## we can have multiple graphs
        for i, name in enumerate(self.names):
            ## if there is a processed file, we skip the self.single_g_process()
            if os.path.exists(self.processed_paths[i]):
                logging.info(f"Found process file of {name} in {self.processed_paths[i]}, skipping process()")
                continue 
                        
            data_lens_for_split.append(
                self.single_g_process(i)
            )

    @property
    def raw_file_names(self):
        raw_file_names = []
        for name in self.names:
            raw_file_names.append(name+'.pt')
        
        return raw_file_names
    
    @property
    def processed_dir(self) -> str:
        if self.task_level == 'edge':
            return os.path.join(self.root, 'processed_for_edges')
        elif self.task_level == 'node':
            return os.path.join(self.root, 'processed_for_nodes')
        else:
            raise ValueError(f"No defination of task {self.task_level}!")

    @property
    def processed_file_names(self):
        processed_names = []
        for i, name in enumerate(self.names):
            if self.sample_rates[i] < 1.0:
                name += f"_s{self.sample_rates[i]}"

            if self.neg_edge_ratio < 1.0:
                name += f"_nr{self.neg_edge_ratio:.1f}"
            processed_names.append(name+"_processed.pt")
        return processed_names

def adaption_for_sgrl(dataset):
    """
    It is only used for contrastive learning (SGRL).
    """
    data_list = []

    for i, name in enumerate(dataset.names):
        single_graph = Data(
            x=dataset[i].node_type, 
            edge_index=dataset[i].edge_index, 
            edge_attr=dataset[i].edge_type
        )
        single_graph.node_attr = dataset[i].node_attr
        data_list.append(single_graph)

    # Create a big graph concate all graphs from the `data_list`
    batch = Batch.from_data_list(data_list)

    ## Change the size of x
    batch.x = batch.x.view(-1, 1)
    ## Rename the `edge_attr` to `edge_type`
    batch.edge_type = batch.edge_attr

    ## remove the redundant attributes
    del batch.edge_attr

    print("attributes in big batch", batch)
    print("batch.ptr", batch.ptr)

    return batch

def performat_SramDataset(dataset_dir, name, 
                          neg_edge_ratio, to_undirected, 
                          small_dataset_sample_rates, large_dataset_sample_rates,
                          task_level,
                          net_only,
                          class_boundaries
                          ):
    start = time.perf_counter()
    names = name.split('+')
    sr_list = [
        large_dataset_sample_rates if gname == 'ultra8t' or gname == 'sandwich' else small_dataset_sample_rates 
        for gname in names
    ]

    dataset = SealSramDataset(
            name=name, root=dataset_dir,
            neg_edge_ratio=neg_edge_ratio,
            to_undirected=to_undirected,
            sample_rates=sr_list,
            task_level=task_level,              
            net_only=net_only,
            class_boundaries=class_boundaries
        )

    elapsed = time.perf_counter() - start
    timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) \
            + f'{elapsed:.2f}'[-3:]
    print(f"PID = {os.getpid()}")
    print(f"Building dataset {name} took {timestr}")
    print('Dataloader: Loading success.')

    return dataset