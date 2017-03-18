# -*- coding:utf-8 -*-
"""
Algorithms about calculate the information entropy of left and right list.
Author:aluka.han
Email:aluka_hxl@gmail.com
Reference:
    https://github.com/Moonshile/ChineseWordSegmentation
    http://www.matrix67.com/blog/archives/5044
    https://zlc1994.com/2017/01/04/
"""
import math


def cal_infor_entropy(_list):
    """
    Given a list of some items, compute entropy of the list
    The entropy is sum of -p[i]*log(p[i]) for every unique element i in the list, and p[i] is its frequency
    :param _list: the list contain all element
    :return: the entropy of the list
    """
    lens = float(len(_list))
    if lens == 0:
        return 0
    else:
        element = {}
        for item in _list:
            element[item] = element.get(item, 0) + 1
        return sum(map(lambda v: -v/lens*math.log(v/lens), element.values()))