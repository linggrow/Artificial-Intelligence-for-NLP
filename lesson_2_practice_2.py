import random
import math
from math import radians, cos, sin, asin, sqrt, acos
import pandas as pd
import numpy as np
import re
from collections import Counter
import jieba
from functools import reduce
from operator import add, mul
import matplotlib.pyplot as plt
# % matplotlib inline
import matplotlib
import networkx as nx
from collections import defaultdict
import requests
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
from sklearn.datasets import load_boston
from geopy import distance
from geopy.geocoders import Nominatim
# print(matplotlib.__path__)


coordination_source = """
{name:'兰州', geoCoord:[103.73, 36.03]},
{name:'嘉峪关', geoCoord:[98.17, 39.47]},
{name:'西宁', geoCoord:[101.74, 36.56]},
{name:'成都', geoCoord:[104.06, 30.67]},
{name:'石家庄', geoCoord:[114.48, 38.03]},
{name:'拉萨', geoCoord:[102.73, 25.04]},
{name:'贵阳', geoCoord:[106.71, 26.57]},
{name:'武汉', geoCoord:[114.31, 30.52]},
{name:'郑州', geoCoord:[113.65, 34.76]},
{name:'济南', geoCoord:[117, 36.65]},
{name:'南京', geoCoord:[118.78, 32.04]},
{name:'合肥', geoCoord:[117.27, 31.86]},
{name:'杭州', geoCoord:[120.19, 30.26]},
{name:'南昌', geoCoord:[115.89, 28.68]},
{name:'福州', geoCoord:[119.3, 26.08]},
{name:'广州', geoCoord:[113.23, 23.16]},
{name:'长沙', geoCoord:[113, 28.21]},
//{name:'海口', geoCoord:[110.35, 20.02]},
{name:'沈阳', geoCoord:[123.38, 41.8]},
{name:'长春', geoCoord:[125.35, 43.88]},
{name:'哈尔滨', geoCoord:[126.63, 45.75]},
{name:'太原', geoCoord:[112.53, 37.87]},
{name:'西安', geoCoord:[108.95, 34.27]},
//{name:'台湾', geoCoord:[121.30, 25.03]},
{name:'北京', geoCoord:[116.46, 39.92]},
{name:'上海', geoCoord:[121.48, 31.22]},
{name:'重庆', geoCoord:[106.54, 29.59]},
{name:'天津', geoCoord:[117.2, 39.13]},
{name:'呼和浩特', geoCoord:[111.65, 40.82]},
{name:'南宁', geoCoord:[108.33, 22.84]},
//{name:'西藏', geoCoord:[91.11, 29.97]},
{name:'银川', geoCoord:[106.27, 38.47]},
{name:'乌鲁木齐', geoCoord:[87.68, 43.77]},
{name:'香港', geoCoord:[114.17, 22.28]},
{name:'澳门', geoCoord:[113.54, 22.19]}
"""

city_location = {
    '香港': (114.17, 22.28)
}

test_string = "{name:'兰州', geoCoord:[103.73, 36.03]},"
pattern = re.compile(r"name:'(\w+)',\s+geoCoord:\[(\d+.\d+),\s(\d+.\d+)\]")

for line in coordination_source.split('\n'):
    city_info = pattern.findall(line)
    if not city_info:
        continue
    city, long, lat = city_info[0]

    long, lat = float(long), float(lat)
    city_location[city] = (long, lat)


print(city_location)

# math的各项公式是怎么计算的，代表的公式长什么样子
# 地球的赤道半径  6378.1公里
# 地球极半径 6358.8公里
# 地球的平均半径为6371千米，赤道半径6378千米，极半径6357千米。赤道周长约为4万千米。

# 地球球面2点之间的距离。
# 返回的是经纬度之间的弧度值。

# 设所求点A ，纬度角β1 ，经度角α1 ；点B ，纬度角β2 ，经度角α2。
# 关于公式有非常错误的地方
# X网上查的距离公式S=R·arc cos[cosβ1cosβ2cos（α1-α2）+ sinβ1sinβ2]，其中R为球体半径
# radius * acos(cos(lat1) * cos(lat2) * cos(lon2 - lon1) + sin(lat1) * sin(lat2))  # 结果是 8148
# X计算速度更快的Google公开的距离计算: $s = 2*asin(sqrt(pow(sin(($radLat1-$radLat2)/2),2)+cos($radLat1)*cos($radLat2)*pow(sin(($radLng1-$radLng2)/2),2)))*$R;
# 2 * radius * asin(sqrt(sin(dlat/2) * sin(dlat/2) + cos(lat1) * cos(lat2) * sin(dlat/2) * sin(dlat/2)))   # 结果是 126
# 百度实际直线测量 163
# 实际运行时，还需要考虑运算的速度，尝试变换代数式或者实现方法来加速
def geo_distance(origin, destination):
    # longitude（经度） and latitude（维度）
    lon1, lat1 = origin
    lon2, lat2 = destination
    radius = 6371

    dlat = math.radians(lat1 - lat2)
    dlon = math.radians(lon1 - lon2)

    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) * math.sin(dlon / 2)
         )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c
    return d

city1 = '上海'
city2 = '杭州'


def get_geo_distance(city1, city2):
    return geo_distance(city_location[city1], city_location[city2])

get_geo_distance('上海', '杭州')
# round(distance(city_location[city1], city_location[city2]), 1)


# matplotlib.use('TkAgg')

city_graph = nx.Graph()
city_graph.add_nodes_from(list(city_location.keys()))

a = 0
nx.draw(city_graph, city_location, with_labels=True, node_size=30)


# dict = defaultdict(factory_function) 这个factory_function可以是list、set、str等
city_connection = defaultdict(list)

threshold = 600
for c1 in city_location:
    for c2 in city_location:
        if c1 == c2:
            continue
        distance = get_geo_distance(c1, c2)
        print(c1, c2, distance)

        if distance < threshold:
            city_connection[c1].append(c2)
            city_connection[c2].append(c1)

# city_connection
a = 0
city_with_road = nx.Graph(city_connection)
city_with_road


nx.draw(city_with_road, city_location, with_labels = True, node_size=30)
simple_connection_info_src = {
    '北京': ['太原', '沈阳'],
    '太原': ['北京', '西安', '郑州'],
    '兰州': ['西安'],
    '郑州': ['太原', '武汉', '长沙', '南昌'],
    '南昌': ['福州', '武汉'],
    '西安': ['兰州', '长沙'],
    '长沙': ['福州', '南宁'],
    '沈阳': ['北京']
}


simple_connection_info = defaultdict(list)
simple_connection_info.update(simple_connection_info_src)

# 下面这段代码是一个理解运行流程的举例
# pop() 函数用于移除列表中的一个元素（默认最后一个元素），并且返回该元素的值。
def bfs(graph, start):
    """
    :param graph:
    :param start:
    :return:
    """
    visited = [start]
    seen = set()
    while visited:
        froninter = visited.pop()
        if froninter in seen:
            continue
        for successor in graph[froninter]:
            if successor in seen:
                continue
            print(successor)
            visited = [successor] + visited

        seen.add(froninter)
    return seen
number_grpah = defaultdict(list)
number_grpah.update({
    1: [2, 3],
    2: [1, 4],
    3: [1, 5],
    4: [2, 6],
    5: [3, 7],
    7: [5, 8]
})


bfs(number_grpah, 1)

# simple_connection_info['西安']
# 注意这里西安只连接了兰州和长沙，但是实际太原里的数据也有西安
nx.draw(nx.Graph(simple_connection_info), city_location, with_labels=True, node_size=10)

# pop(N)移除第N个元素
def search(start, destination, connection_grpah, sort_candidate):
    pathes = [[start]]
    visitied = set()

    # if we find existing pathes
    # 这是队列，把第一个取出，继续搜索节点，如果还存在节点，把节点加再放到队列的末尾
    while pathes:  # if we find existing pathes
        path = pathes.pop(0)
        froninter = path[-1]

        if froninter in visitied:
            continue
        successors = connection_grpah[froninter]

        for city in successors:
            if city in path:
                continue  # eliminate loop
            # 只有还有节点的时候，才能回队列末尾，不然就丢弃
            new_path = path + [city]
            pathes.append(new_path)
            if city == destination:
                ## ？只要搜索到了一条路就马上返回了，显示会有例外吗？节点少，但是距离长。
                return new_path

        visitied.add(froninter)

        # 会对每次可能的路径排序，优先最短的路径先取出处理
        pathes = sort_candidate(pathes)  # 我们可以加一个排序函数 对我们的搜索策略进行控制

# 节点最少
def transfer_stations_first(pathes):
    return sorted(pathes, key=len)
# 节点最多
def transfer_as_much_possible(pathes):
    return sorted(pathes, key=len, reverse=True)

# 距离最短
def shortest_path_first(pathes):
    if len(pathes) <= 1:
        return pathes

    def get_path_distnace(path):
        distance = 0
        # 每一个地点与最后一个地点的距离之和的排序。暂时。
        for station in path[:-1]:
            distance += get_geo_distance(station, path[-1])
        return distance

    return sorted(pathes, key=get_path_distnace)

search('兰州', '福州', simple_connection_info, sort_candidate=shortest_path_first)

def pretty_print(cities):
    try:
        print('🚗->'.join(cities))
    except:
        # 当路径为450时，路径是为空
        print('路径未找到，或数据出错')


pretty_print(search('北京', '福州', simple_connection_info, sort_candidate=shortest_path_first))
pretty_print(search('北京', '南京', city_connection, sort_candidate=transfer_stations_first))
pretty_print(search('北京', '广州', city_connection, sort_candidate=transfer_as_much_possible))


# 为何总是会出现：
# pretty_print(search('北京', '南京', city_connection))，
# TypeError: search() missing 1 required positional argument: 'sort_candidate'
# 问题：通常调用函数时，参数都可以默认。为什么这里一定要填，而如何才能不用填。
