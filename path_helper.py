#!/usr/lib/env python
#coding:utf-8
"""
  description:
    去掉路径数组中包含的无效路径
    dirs： 表示的是输入的目录数组
    args:  表示的是需要删除的可变数目的待删除文件名，可以删除多个
"""
def remove_file_in_dirs(dirs,*args):
    for file_delete in args:
        if file_delete in dirs:
            dirs.remove(file_delete)
    return dirs
