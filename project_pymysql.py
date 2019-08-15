import pymysql
import pandas as pd
import time
filename = "/Users/lingrowzhang/Documents/Artificial-Intelligence-for-NLP/date/"
# 错误方式示范
# data = []
# with open(filename + 'news_chinese.csv', 'r') as f:
#     for a in f.readlines():
#         data.append(a.split("','"))
# data_table = pd.DataFrame(data = data)
# data = pd.read_csv(filename + "news_chinese.csv")

db = pymysql.connect("rm-8vbwj6507z6465505ro.mysql.zhangbei.rds.aliyuncs.com",
                     "root",
                     "AI@2019@ai",
                     "stu_db")
# 表名：news_chinese
cursor = db.cursor()
sql_test = """
        SELECT id, author, source, content, feature, title, url
        FROM news_chinese WHERE id < 10
        """
sql = """
        SELECT id, author, source, content, feature, title, url
        FROM news_chinese 
        """
a = cursor.execute(sql)
result = cursor.fetchall()
data_list = []
for the_result in list(result):
    data_list.append(list(the_result))

data_table = pd.DataFrame(data=data_list)

data_table.to_csv(filename + "news_chinese.csv", index=False)
data_2 = pd.read_csv(filename + "news_chinese.csv")


# SQL 删除语句
# sql = "DELETE FROM EMPLOYEE WHERE AGE > %s" % (20)
try:
   # 执行SQL语句
   cursor.execute(sql)
   # 提交修改
   db.commit()
except:
   # 发生错误时回滚
   db.rollback()