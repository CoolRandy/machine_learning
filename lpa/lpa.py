#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
    lpa算法原理：
    该代码采用networkx中提供的默认图：图的节点总共包含34个，具体的链接参见图示
    
    该实现存在两个问题：
    1、本样例没有考虑节点的权重以及边的权重
    2、随机选择一个节点查找相邻节点，默认是将该节点排除在外的
    
    上述问题会导致，每次执行会导致划分的子图是变化的，也就是说振荡很大，为保证尽可能小的振荡，需要解决上述两个问题
    
"""

import collections
import random
import networkx as nx
import matplotlib.pyplot as plt


class LPA():
    def __init__(self, G, max_iter=20):
        self._G = G
        self._n = len(G.node)
        self._max_iter = max_iter
        print self._n
    
    def can_stop(self):
        # all node has the label same with its most neighbor
        for i in range(self._n):
            print '============'
            node = self._G.node[i]
            print node
            label = node['label']
            print 'label: ', label
            max_labels = self.get_max_neighbor_label(i)
            print 'max label: ', max_labels
            if label not in max_labels:
                return False
        return True
    
    def get_max_neighbor_label(self, node_index):
        print '**************'
        m = collections.defaultdict(int)
        print self._G.neighbors(node_index)
        for neighbor_index in self._G.neighbors(node_index):
            neighbor_label = self._G.node[neighbor_index]['label']
            m[neighbor_label] += 1
        print m
        max_v = max(m.itervalues())
        print 'max_v: ', max_v
        return [item[0] for item in m.items() if item[1] == max_v]
    
    def populate_label(self):
        # random visit
        print 'random visit'
        # 随机34个
        visitSequence = random.sample(self._G.nodes(), len(self._G.nodes()))
        print 'visit sequence: ', visitSequence
        # 循环34次
        for i in visitSequence:
            node = self._G.node[i]
            label = node['label']
            print 'visit label: ', label
            max_labels = self.get_max_neighbor_label(i)
            print 'visit max labels: ', max_labels
            if label not in max_labels:
                # 如果标签不在max_labels中，随机选择一个节点更新当前的节点
                newLabel = random.choice(max_labels)
                node['label'] = newLabel

def get_communities(self):
    communities = collections.defaultdict(lambda: list())
        for node in self._G.nodes(True):
            print 'node: ', node
            label = node[1]["label"]
            communities[label].append(node[0])
    print communities
        return communities.values()

def execute(self):
    # 初始化节点一个唯一的标签，这里采用id
    for i in range(self._n):
        self._G.node[i]['label'] = i
            print self._G.node[i]
        
        iter_time = 0
        while (not self.can_stop() and iter_time < self._max_iter):
            self.populate_label()
            iter_time += 1
        return self.get_communities()


if __name__ == '__main__':
    G = nx.karate_club_graph()
    # print G
    # 打印每个节点的度
    print 'Node Degree: '
    for v in G:
        print('%s %s' % (v, G.degree(v)))
    # nx.draw_circular(G, with_labels=True)
    # plt.show()
    
    algorithm = LPA(G)
    communities = algorithm.execute()
    for community in communities:
        print community
