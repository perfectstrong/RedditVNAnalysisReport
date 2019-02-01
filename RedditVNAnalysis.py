# -*- coding: utf-8 -*-
#%%
######################################################
# libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import AxesGrid
import math
from scipy import stats

#%%
######################################################
# import data and filter
infile = "20170101T000000-20190101T000000.csv"
df = pd.read_csv(infile, encoding="utf-8", index_col=None, header=0, lineterminator="\n")
totalUnfiltered = df.shape[0]
print("Số bài viết tổng cộng: {:d}".format(totalUnfiltered))
df.drop(df[df["r_created_utc"] == 0].index, inplace=True) # filter
totalFiltered = df.shape[0]
print("Số bài viết sau khi lọc: {:d}. Tương đương {:.2f}% tổng số bài.".format(totalFiltered, totalFiltered / totalUnfiltered * 100))

######################################################
# define columns types
df["r"] = df["r"].astype("category")
df["user_raw_id"] = df["user_raw_id"].astype(str)
# avoid case sensitive
df["r"] = df["r"].str.lower()
# time conversion
df["created_time"]=pd.to_datetime(df["created_time"], format="%Y-%m-%dT%H:%M:%S.000Z", utc=True).dt.tz_convert("Asia/Ho_Chi_Minh")
df["r_created_utc"]=pd.to_datetime(df["r_created_utc"], unit="s", utc=True).dt.tz_convert("Asia/Ho_Chi_Minh")

######################################################
# determine week day of post
vietnameseDaysOfWeek = ['Hai', 'Ba', 'Tư', 'Năm', 'Sáu', 'Bảy', 'CN']
df["weekday"] = df["created_time"].dt.weekday
df["weekday"] = df["weekday"].astype("category")
r = pd.crosstab(index=df["r"], columns="count")
r.sort_values(by="count", ascending=False, inplace=True)
######################################################
# prepare hour stats
Nrow = 2
Ncol = 4
Noffset = 0
numWeeks = math.ceil((df.iloc[-1]["created_time"] - df.iloc[0]["created_time"]).days/7)
def meanPostsPerWeek(series):
    return np.size(series) / numWeeks
listTopSubs = r.index[Noffset:(Noffset+Nrow*Ncol)].tolist()
dfWeekdayHourPerSub = df.pivot_table(index=["r", df["created_time"].dt.hour], columns="weekday", values=["likes_count", "post_id"], aggfunc={"likes_count": np.mean, "post_id": meanPostsPerWeek}, fill_value=0)

######################################################
# translators stats
translators = df[["user_raw_id", "user_name"]]
translators.drop_duplicates(subset="user_raw_id", keep='first', inplace=True)
translators.set_index("user_raw_id", inplace=True)
translatorsStats = df.pivot_table(index="user_raw_id", values=["likes_count", "post_id"], aggfunc={"likes_count": [np.sum, np.mean], "post_id": [np.size, meanPostsPerWeek]}, fill_value=0)

#%%
######################################################
# retrospect memory usage
df.info(memory_usage='deep')
for dtype in list(set(df.dtypes)):
    selected_dtype = df.select_dtypes(include=[dtype])
    sum_usage_b = selected_dtype.memory_usage(deep=True).sum()
    sum_usage_mb = sum_usage_b / 1024 ** 2
    print("Sum memory usage for {} columns: {:03.2f} MB".format(dtype,sum_usage_mb))
print("Usage of each column in MB")
for colName, usageB in df.memory_usage(index=True, deep=True).items():
    print("{:<20} {:10.2f} MB".format(colName, usageB / 1024 ** 2))

#%%
######################################################
# posts by month
df["created_time_utc"] = df["created_time"].dt.tz_convert('UTC')
grouperAllByMonth = df.groupby(by=pd.Grouper(freq="M", key="created_time_utc"))
allPostsByMonth = grouperAllByMonth["post_id"].size().fillna(0)
allLikesByMonth = grouperAllByMonth["likes_count"].sum().fillna(0)

fig = plt.figure()
plt.title("Thống kê tổng số post và like", fontsize=24)
monthLabels = allPostsByMonth.index.strftime("%m-%Y").tolist()
ax1 = fig.add_subplot(111)
ax1.plot(monthLabels, allPostsByMonth, "r-", label="Post")
ax1.set_xlabel("Tháng", fontsize=16)
ax1.set_ylabel("Số bài viết", fontsize=16)
ax1.grid(True)
ax2 = ax1.twinx()
ax2.plot(monthLabels, allLikesByMonth, "b-", label="Like")
ax2.set_ylabel("Số like", fontsize=16)
fig.tight_layout()
fig.autofmt_xdate()
plt.tick_params(axis="both", labelsize=14)
handles,labels = [],[]
for ax in fig.axes:
    for h,l in zip(*ax.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)
plt.legend(handles, labels, loc="upper left")
plt.show()

#%%
######################################################
# hot subreddits
subs = r.index.nunique()
print("Số subreddit được dịch: {:d}".format(subs))
print("Số bài dịch mỗi sub trung bình: {:f}".format(float(r.mean())))
subsGreater1 = (r > 1).sum()[0]
print("Số subreddit có trên 1 bài viết: {:d}. Tương đương {:.2f}% tổng số sub.".format(subsGreater1, subsGreater1 / subs * 100))
subsGreater10 = (r > 10).sum()[0]
print("Số subreddit có trên 10 bài viết: {:d}. Tương đương {:.2f}% tổng số sub.".format(subsGreater10, subsGreater10 / subs * 100))
subsGreater100 = (r > 100).sum()[0]
print("Số subreddit có trên 100 bài viết: {:d}. Tương đương {:.2f}% tổng số sub.".format(subsGreater100, subsGreater100 / subs * 100))

#%%
######################################################
# top translated subreddits
N = 30
rTop = r.head(N)
y = np.arange(N)

fig, ax = plt.subplots()
plt.barh(y, rTop["count"])
plt.title("Những Subreddit nhiều bài dịch nhất", fontsize=24)
plt.xlabel("Số bài được dịch", fontsize=18)
plt.yticks(y, rTop.index.tolist(), fontsize=16)
for i in ax.patches:
    ax.text(i.get_width() + .1, i.get_y() + 0.5, str(int(i.get_width())), fontsize=16, color="black")
ax.invert_yaxis()
plt.tight_layout()
plt.show()

#%%
######################################################
# top interested subreddits
interests = df.groupby("r")["likes_count", "comments_count"].sum()
interests["sum"] = interests["likes_count"] + interests["comments_count"]
interests.sort_values(by="sum", ascending=False, inplace=True)
Ni = 30
yi = np.arange(Ni)
ax = interests.iloc[0:Ni][["likes_count","comments_count"]].plot.barh(stacked=True)
plt.title("Những Subreddit được chú ý nhất", fontsize=24)
plt.xlabel("Tổng số cảm xúc và bình luận", fontsize=18)
plt.yticks(yi, rTop.index.tolist(), fontsize=16)
plt.ylabel("")
plt.tight_layout()
for i, subreddit in enumerate(interests[0:Ni].index.tolist()):
    s = interests.loc[subreddit]["sum"]
    ax.annotate(str(s), (s, i + .1), fontsize=16, color="black")
ax.invert_yaxis()

#%%
######################################################
# askreddit's comments
ratioCommentsCount = interests["comments_count"][0] / interests["comments_count"]
ratioCommentsCount = ratioCommentsCount.drop("askreddit")
ratioCommentsCount = ratioCommentsCount.replace(np.inf, np.nan).dropna()
print("askreddit có số comment nhiều hơn các sub khác {:.2f} lần trở lên.".format(float(ratioCommentsCount.min())))

#%%
######################################################
# mean interests
meanInterests = df.groupby("r")["likes_count", "comments_count"].mean()
for i in range(Ni):
    subreddit = interests.index[i]
    print("Trung bình một bài {:s} sẽ có {:.0f} like (cảm xúc) và {:.0f} bình luận.".format(subreddit, meanInterests.loc[subreddit]["likes_count"], meanInterests.loc[subreddit]["comments_count"]))

print("Còn trung bình toàn thể là {:.0f} like (cảm xúc) và {:.0f} bình luận.".format(df["likes_count"].mean(), df["comments_count"].mean()))

#%%
######################################################
# posts per month overall
grouperByRMonth = df.groupby(["r", pd.Grouper(freq="M", key="created_time")])
countRByMonth = grouperByRMonth["post_id"].count().unstack("r").fillna(0)
Nr = 15
fig, ax = plt.subplots()
plt.set_cmap("nipy_spectral")
monthLabels = countRByMonth.index.strftime("%m-%Y").tolist()
for subreddit in r.index[0:Nr-1].tolist():
    ax.plot(monthLabels, countRByMonth[subreddit].tolist(), label=subreddit)
plt.title("Số bài dịch mỗi sub theo thời gian", fontsize=24)
plt.xticks(fontsize=14)
plt.ylabel("")
plt.xlabel("")
plt.tight_layout()
fig.autofmt_xdate()
ax.legend()
ax.grid(True)
plt.show()

#%%
######################################################
# mean likes per hour overall
dfMeanLikesWeekdayHour = df.pivot_table(index=df["created_time"].dt.hour, columns="weekday", values="likes_count", aggfunc="mean")
fig, ax = plt.subplots()
im = ax.imshow(dfMeanLikesWeekdayHour, cmap="Reds", aspect="auto")
ax.set_xticks(np.arange(len(vietnameseDaysOfWeek)))
ax.set_xticklabels(vietnameseDaysOfWeek)
plt.title("Like trung bình theo giờ đăng", fontsize=24)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel("Giờ", fontsize=16)
plt.xlabel("Thứ", fontsize=16)
plt.tight_layout()
plt.colorbar(im, ax=ax)
plt.show()

#%%
######################################################
# mean posts per hour overall
dfMeanPostsWeekdayHour = df.pivot_table(index=df["created_time"].dt.hour, columns="weekday", values="post_id", aggfunc=np.size)/numWeeks
fig, ax = plt.subplots()
im = ax.imshow(dfMeanPostsWeekdayHour, cmap="Reds", aspect="auto")
ax.set_xticks(np.arange(len(vietnameseDaysOfWeek)))
ax.set_xticklabels(vietnameseDaysOfWeek)
plt.title("Số bài trung bình theo giờ đăng", fontsize=24)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel("Giờ", fontsize=16)
plt.xlabel("Thứ", fontsize=16)
plt.tight_layout()
plt.colorbar(im, ax=ax)
plt.show()

#%%
######################################################
# mean likes per hour per sub
fig = plt.figure()
grid = AxesGrid(fig, 111, nrows_ncols=(Nrow, Ncol), axes_pad=0.5, share_all=True, label_mode="L", cbar_location="right", cbar_mode="single", cbar_pad=0.1, aspect=False)
for i in range(Nrow*Ncol):
    subreddit = listTopSubs[i]
    imTemp = grid[i].imshow(dfWeekdayHourPerSub.query('r == "{:s}"'.format(subreddit))["likes_count"], cmap="Reds", aspect="auto")
    grid[i].set_title(subreddit, fontsize=16)
    grid[i].set_aspect("auto")
    grid[i].set_xlabel("Thứ", fontsize=16)
    grid[i].set_ylabel("Giờ", fontsize=16)
    grid[i].tick_params(axis="both", labelsize=12)
grid.cbar_axes[0].colorbar(imTemp)
grid.axes_llc.set_xticks(np.arange(len(vietnameseDaysOfWeek)))
grid.axes_llc.set_xticklabels(vietnameseDaysOfWeek)
fig.suptitle("Like trung bình mỗi giờ của {:d} sub nổi nhất".format(Nrow*Ncol), fontsize=24)
plt.show()

#%%
######################################################
# mean posts per hour per sub
fig = plt.figure()
grid = AxesGrid(fig, 111, nrows_ncols=(Nrow, Ncol), axes_pad=0.5, share_all=True, label_mode="L", cbar_location="right", cbar_mode="single", cbar_pad=0.1, aspect=False)
for i in range(Nrow*Ncol):
    subreddit = listTopSubs[i]
    imTemp = grid[i].imshow(dfWeekdayHourPerSub.query('r == "{:s}"'.format(subreddit))["post_id"], cmap="Reds", aspect="auto")
    grid[i].set_title(subreddit, fontsize=16)
    grid[i].set_aspect("auto")
    grid[i].set_xlabel("Thứ", fontsize=16)
    grid[i].set_ylabel("Giờ", fontsize=16)
    grid[i].tick_params(axis="both", labelsize=12)
grid.cbar_axes[0].colorbar(imTemp)
grid.axes_llc.set_xticks(np.arange(len(vietnameseDaysOfWeek)))
grid.axes_llc.set_xticklabels(vietnameseDaysOfWeek)
fig.suptitle("Số post trung bình mỗi giờ của {:d} sub nổi nhất".format(Nrow*Ncol), fontsize=24)
plt.show()

#%%
######################################################
# statistics of likes of posts
print("Tổng số like: {:d}.".format(df["likes_count"].sum()))
print("Số like trung bình mỗi post: {:.2f}".format(df["likes_count"].mean()))
print("Số like cao nhất: {:d}".format(df["likes_count"].max()))
print("Bài nhiều like nhất: https://www.facebook.com/groups/redditvietnam/permalink/{}/".format(df.loc[df["likes_count"].idxmax()]["post_raw_id"]))
#likescountQuantiles = [.1, .25, .5, .75, .9, .99]
#for i in likescountQuantiles:
#    print("{:.0%} bài có trên {:.0f} like.".format(1 - i, df["likes_count"].quantile(i)))
likescountMarkpoints = [100, 300, 500, 1000, 2000, 4000, 6000, 8000]
for i in likescountMarkpoints:
    p = (df["likes_count"] >= i).sum()
    print("Có {:d} bài ({:.2%}) đạt {:d} like trở lên.".format(p, p / totalFiltered, i))
fig, ax = plt.subplots()
likescountBins = range(0, 4000, 100)
df.hist(column="likes_count", grid=True, xlabelsize=14, ylabelsize=14, bins=likescountBins, ax=ax)
plt.title("Phân bố số like", fontsize=24)
plt.xlabel("Số like", fontsize=16)
plt.ylabel("Số bài viết", fontsize=16)
plt.show()

#%%
######################################################
# estimate distribution
sc = 500
reducedLikescount = df["likes_count"].copy() / sc
l = np.arange(0, 4000 / sc, 100 / (sc * 2))
param = stats.lognorm.fit(reducedLikescount)
pdfFitted = stats.lognorm.pdf(l, param[0], param[1], param[2])
fig = plt.figure()
plt.hist(reducedLikescount, bins=l, density=True)
plt.plot(l, pdfFitted, "r-")
plt.title("Ước lượng phân bố", fontsize=24)
plt.xlabel("Số bài viết (đơn vị: {:d} bài)".format(sc), fontsize=16)
plt.ylabel("Mật độ", fontsize=16)
kstest = stats.kstest(reducedLikescount, 'lognorm', param)
plt.text(7, 0.5, "Model: lognorm\nShape = {:f}\nLoc = {:f}\nScale = {:f}\nKS-test:\nD = {:f}\np-value: {:f}".format(param[0], param[1], param[2], kstest.statistic, kstest.pvalue), fontsize=16)
plt.show()

#%%
######################################################
# statistics of translators
print("Tổng số dịch giả: {:d}".format(translators.size))
print("Mỗi dịch giả trung bình dịch {:.0f} bài.".format(totalFiltered / translators.size))
print("Dịch giả chăm chỉ nhất: {:s} (https://facebook.com/{:d}) với {:d} bài.".format(translators.loc[translatorsStats[('post_id', 'size')].idxmax()]["user_name"], translatorsStats[('post_id', 'size')].idxmax(), translatorsStats[('post_id', 'size')].max()))
print("Dịch giả dễ thương nhất: {:s} (https://facebook.com/{:d}) với tổng cộng {:d} like.".format(translators.loc[translatorsStats[('likes_count', 'sum')].idxmax()]["user_name"], translatorsStats[('likes_count', 'sum')].idxmax(), translatorsStats[('likes_count', 'sum')].max()))
print("Dịch giả hay được cưng yêu nhất: {:s} (https://facebook.com/{:d}) với trung bình {:.0f} like mỗi bài.".format(translators.loc[translatorsStats[('likes_count', 'mean')].idxmax()]["user_name"], translatorsStats[('likes_count', 'mean')].idxmax(), translatorsStats[('likes_count', 'mean')].max()))
postscountMarkpoints = [10, 20, 50, 100, 200]
for i in postscountMarkpoints:
    p = translatorsStats[translatorsStats[("post_id", "size")] >= i].count()[0]
    print("{:d} dịch giả ({:.2%}) có {:d} bài dịch trở lên.".format(p, p / translators.size, i))

#%%
######################################################
# reddit posts verification
patternStr = r"https?://www\.reddit\.com/r/\w+/comments/(\w{1,6})/|https?://redd\.it/(\w{1,6})"
dfLinks = df["message"].str.extractall(patternStr)
dfLinksList = dfLinks.groupby(level=0)[0].apply(list) + dfLinks.groupby(level=0)[1].apply(list)
del dfLinks
dfLinksListCount = dfLinksList.apply(lambda x: len(set([y for y in x if str(y) != 'nan'])))
multilinksSubmissionCount = (dfLinksListCount > 1).sum()
print("Số bài dịch có 2 link reddit trở lên trong bài là {:d}, chiếm {:.2%} tổng số bài.".format(multilinksSubmissionCount, multilinksSubmissionCount / totalFiltered))
