# -*- coding: utf-8 -*-
#%%
######################################################
# libraries
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import AxesGrid
import math
from scipy import stats

# matplotlib params
mpl.rcParams["axes.titlesize"] = 24
mpl.rcParams["axes.grid"] = True
mpl.rcParams["axes.labelsize"] = 16
mpl.rcParams["xtick.labelsize"] = 14
mpl.rcParams["ytick.labelsize"] = 14

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
# interests mesure
interests = df.groupby("r")["likes_count", "comments_count"].sum()
interests["sum"] = interests["likes_count"] + interests["comments_count"]
interests.sort_values(by="sum", ascending=False, inplace=True)

######################################################
# prepare hour stats
numWeeks = math.ceil((df.iloc[-1]["created_time"] - df.iloc[0]["created_time"]).days/7)
def meanPostsPerWeek(series):
    return np.size(series) / numWeeks
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
    sumUsageB = selected_dtype.memory_usage(deep=True).sum()
    sumUsageMB = sumUsageB / 1024 ** 2
    print("Sum memory usage for {} columns: {:03.2f} MB".format(dtype,sumUsageMB))
print("Usage of each column in MB")
for colName, usageB in df.memory_usage(index=True, deep=True).items():
    print("{:<20} {:10.2f} MB".format(colName, usageB / 1024 ** 2))

del dtype, selected_dtype, sumUsageB, sumUsageMB, colName, usageB

#%%
######################################################
# posts by month
df["created_time_utc"] = df["created_time"].dt.tz_convert('UTC')
grouperAllByMonth = df.groupby(by=pd.Grouper(freq="M", key="created_time_utc"))
allPostsByMonth = grouperAllByMonth["post_id"].size().fillna(0)
allLikesByMonth = grouperAllByMonth["likes_count"].sum().fillna(0)

fig = plt.figure()
plt.title("Thống kê tổng số post và like")
monthLabels = allPostsByMonth.index.strftime("%m-%Y").tolist()
ax1 = fig.add_subplot(111)
ax1.plot(monthLabels, allPostsByMonth, "r-", label="Post")
ax1.set_xlabel("Tháng")
ax1.set_ylabel("Số bài viết")
ax2 = ax1.twinx()
ax2.plot(monthLabels, allLikesByMonth, "b-", label="Like")
ax2.set_ylabel("Số like")
fig.autofmt_xdate()
handles, labels = [],[]
for ax in fig.axes:
    for h, l in zip(*ax.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)
plt.legend(handles, labels, loc="upper left")
plt.show()

del grouperAllByMonth, allPostsByMonth, allLikesByMonth, handles, labels, h, l, fig, ax1, ax2
df.drop("created_time_utc", axis=1, inplace=True)

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

del subs, subsGreater1, subsGreater10, subsGreater100

#%%
######################################################
# top translated subreddits
N = 30
rTop = r.head(N)
y = np.arange(N)

fig, ax = plt.subplots()
plt.barh(y, rTop["count"])
plt.title("Những Subreddit nhiều bài dịch nhất")
plt.xlabel("Số bài được dịch")
plt.yticks(y, rTop.index.tolist())
for i in ax.patches:
    ax.text(i.get_width() + .1, i.get_y() + 0.5, str(int(i.get_width())), color="black")
ax.invert_yaxis()
ax.grid(False)
plt.show()

del N, rTop, y, i, fig, ax

#%%
######################################################
# top interested subreddits
Ni = 30
yi = np.arange(Ni)
ax = interests.iloc[0:Ni][["likes_count","comments_count"]].plot.barh(stacked=True)
plt.title("Những Subreddit được chú ý nhất")
plt.xlabel("Tổng số cảm xúc và bình luận")
plt.yticks(yi, r.head(Ni).index.tolist())
plt.ylabel("")
for i, subreddit in enumerate(interests[0:Ni].index.tolist()):
    s = interests.loc[subreddit]["sum"]
    ax.annotate(str(s), (s, i + .1), color="black")
ax.invert_yaxis()
ax.grid(False)

del Ni, yi, ax, i, subreddit, s

#%%
######################################################
# askreddit's comments
ratioCommentsCount = interests["comments_count"][0] / interests["comments_count"]
ratioCommentsCount = ratioCommentsCount.drop("askreddit")
ratioCommentsCount = ratioCommentsCount.replace(np.inf, np.nan).dropna()
print("askreddit có số comment nhiều hơn các sub khác {:.2f} lần trở lên.".format(float(ratioCommentsCount.min())))

del ratioCommentsCount

#%%
######################################################
# mean interests
meanInterests = df.groupby("r")["likes_count", "comments_count"].mean()
N = 30
for i in range(N):
    subreddit = interests.index[i]
    print("Trung bình một bài {:s} sẽ có {:.0f} like (cảm xúc) và {:.0f} bình luận.".format(subreddit, meanInterests.loc[subreddit]["likes_count"], meanInterests.loc[subreddit]["comments_count"]))

print("Còn trung bình toàn thể là {:.0f} like (cảm xúc) và {:.0f} bình luận.".format(df["likes_count"].mean(), df["comments_count"].mean()))

del meanInterests, N, i, subreddit

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
plt.title("Số bài dịch mỗi sub theo thời gian")
plt.ylabel("")
plt.xlabel("")
fig.autofmt_xdate()
ax.legend()
plt.show()

del grouperByRMonth, countRByMonth, Nr, fig, ax, monthLabels, subreddit

#%%
######################################################
# mean likes per hour overall
dfMeanLikesWeekdayHour = df.pivot_table(index=df["created_time"].dt.hour, columns="weekday", values="likes_count", aggfunc="mean")
fig, ax = plt.subplots()
im = ax.imshow(dfMeanLikesWeekdayHour, cmap="Reds", aspect="auto")
ax.set_xticks(np.arange(len(vietnameseDaysOfWeek)))
ax.set_xticklabels(vietnameseDaysOfWeek)
plt.title("Like trung bình theo giờ đăng")
plt.ylabel("Giờ")
plt.xlabel("Thứ")
plt.colorbar(im, ax=ax)
plt.show()

del dfMeanLikesWeekdayHour, fig, ax, im

#%%
######################################################
# mean posts per hour overall
dfMeanPostsWeekdayHour = df.pivot_table(index=df["created_time"].dt.hour, columns="weekday", values="post_id", aggfunc=np.size)/numWeeks
fig, ax = plt.subplots()
im = ax.imshow(dfMeanPostsWeekdayHour, cmap="Reds", aspect="auto")
ax.set_xticks(np.arange(len(vietnameseDaysOfWeek)))
ax.set_xticklabels(vietnameseDaysOfWeek)
plt.title("Số bài trung bình theo giờ đăng")
plt.ylabel("Giờ")
plt.xlabel("Thứ")
plt.colorbar(im, ax=ax)
plt.show()

del dfMeanPostsWeekdayHour, fig, ax, im

#%%
######################################################
# mean likes per hour per sub
Nrow = 2
Ncol = 4
Noffset = 0
listTopSubs = r.index[Noffset:(Noffset+Nrow*Ncol)].tolist()
fig = plt.figure()
grid = AxesGrid(fig, 111, nrows_ncols=(Nrow, Ncol), axes_pad=0.5, share_all=True, label_mode="L", cbar_location="right", cbar_mode="single", cbar_pad=0.1, aspect=False)
for i in range(Nrow*Ncol):
    subreddit = listTopSubs[i]
    imTemp = grid[i].imshow(dfWeekdayHourPerSub.query('r == "{:s}"'.format(subreddit))["likes_count"], cmap="Reds", aspect="auto")
    grid[i].set_title(subreddit, fontsize=20)
    grid[i].set_aspect("auto")
    grid[i].set_xlabel("Thứ")
    grid[i].set_ylabel("Giờ")
    grid[i].tick_params(axis="both", labelsize=12)
    grid[i].grid(False)
grid.cbar_axes[0].colorbar(imTemp)
grid.axes_llc.set_xticks(np.arange(len(vietnameseDaysOfWeek)))
grid.axes_llc.set_xticklabels(vietnameseDaysOfWeek)
fig.suptitle("Like trung bình mỗi giờ của {:d} sub nổi nhất".format(Nrow*Ncol), fontsize=24)
plt.show()

del Nrow, Ncol, Noffset, listTopSubs, fig, grid, i, subreddit, imTemp

#%%
######################################################
# mean posts per hour per sub
Nrow = 2
Ncol = 4
Noffset = 0
listTopSubs = r.index[Noffset:(Noffset+Nrow*Ncol)].tolist()

fig = plt.figure()
grid = AxesGrid(fig, 111, nrows_ncols=(Nrow, Ncol), axes_pad=0.5, share_all=True, label_mode="L", cbar_location="right", cbar_mode="single", cbar_pad=0.1, aspect=False)
for i in range(Nrow*Ncol):
    subreddit = listTopSubs[i]
    imTemp = grid[i].imshow(dfWeekdayHourPerSub.query('r == "{:s}"'.format(subreddit))["post_id"], cmap="Reds", aspect="auto")
    grid[i].set_title(subreddit, fontsize=20)
    grid[i].set_aspect("auto")
    grid[i].set_xlabel("Thứ")
    grid[i].set_ylabel("Giờ")
    grid[i].tick_params(axis="both", labelsize=12)
    grid[i].grid(False)
grid.cbar_axes[0].colorbar(imTemp)
grid.axes_llc.set_xticks(np.arange(len(vietnameseDaysOfWeek)))
grid.axes_llc.set_xticklabels(vietnameseDaysOfWeek)
fig.suptitle("Số post trung bình mỗi giờ của {:d} sub nổi nhất".format(Nrow*Ncol), fontsize=24)
plt.show()

del Nrow, Ncol, Noffset, listTopSubs, fig, grid, i, subreddit, imTemp

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
plt.title("Phân bố số like")
plt.xlabel("Số like")
plt.ylabel("Số bài viết")
plt.show()

del likescountMarkpoints, i, p, fig, ax, likescountBins

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
plt.title("Ước lượng phân bố")
plt.xlabel("Số bài viết (đơn vị: {:d} bài)".format(sc))
plt.ylabel("Mật độ")
kstest = stats.kstest(reducedLikescount, 'lognorm', param)
plt.text(7, 0.5, "Model: lognorm\nShape = {:f}\nLoc = {:f}\nScale = {:f}\nKS-test:\nD = {:f}\np-value: {:f}".format(param[0], param[1], param[2], kstest.statistic, kstest.pvalue), fontsize=16)
plt.show()

del sc, reducedLikescount, l, param, pdfFitted, fig, kstest

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
    p = (translatorsStats[("post_id", "size")] >= i).sum()
    print("{:d} dịch giả ({:.2%}) có {:d} bài dịch trở lên.".format(p, p / translators.size, i))

del postscountMarkpoints, i, p

#%%
######################################################
# reddit posts verification
patternStr = r"https?://www\.reddit\.com/r/\w+/comments/(\w{1,6})/|https?://redd\.it/(\w{1,6})"
dfLinks = df["message"].str.extractall(patternStr)
dfLinksList = dfLinks.groupby(level=0)[0].apply(list) + dfLinks.groupby(level=0)[1].apply(list)
dfLinksListCount = dfLinksList.apply(lambda x: len(set([y for y in x if str(y) != 'nan'])))
multilinksSubmissionCount = (dfLinksListCount > 1).sum()
print("Số bài dịch có 2 link reddit trở lên trong bài là {:d}, chiếm {:.2%} tổng số bài.".format(multilinksSubmissionCount, multilinksSubmissionCount / totalFiltered))

del patternStr, dfLinks, dfLinksList, dfLinksListCount, multilinksSubmissionCount

#%%
######################################################
# ratio of karma and comments on submissions
meanKarma, meanCommentsCount = df[["r_score", "r_num_comments"]].mean()
print("Karma trung bình: {:.0f}.".format(meanKarma))
print("Số bình luận trung bình: {:.0f}.".format(meanCommentsCount))
print("Tỉ lệ Karma trên bình luận: {:.2f}.".format(meanKarma / meanCommentsCount))

del meanKarma, meanCommentsCount

#%%
######################################################
# choice of submission and interest reception
dfChoices = df[["r", "r_score", "likes_count", "created_time", "r_created_utc"]].copy()
dfChoices["delta"] = dfChoices["created_time"] - dfChoices["r_created_utc"]
# minor filter
irr = (dfChoices["delta"].dt.days < 0).sum()
print("{:d} bài có delta < 0, chiếm {:.2%} tổng số.".format(irr, irr / totalFiltered))
dfChoices.drop(dfChoices[dfChoices["delta"].dt.days < 0].index, inplace=True)
# convert delta to days in float
dfChoices["days_f"] = dfChoices["delta"] / np.timedelta64(1, "D")
# general stats
print("Khoảng cách xa nhất: {}. Link: https://www.facebook.com/groups/redditvietnam/permalink/{}.".format(dfChoices["delta"].max(), df.loc[dfChoices["delta"].idxmax()]["post_raw_id"]))
delays = [1, 7, 14, 30, 90, 180, 360, 720, 1080, 1800]
sampleSize = len(dfChoices.index)
for i in delays:
    p = (dfChoices["days_f"] >= i).sum()
    print("{:d} submission ({:.2%}) được dịch sau {:d} ngày.".format(p, p / sampleSize, i))
# crop
rScoreCrop = 100000
daysFCrop = 7
dfChoicesCrop = dfChoices[(dfChoices["r_score"] < rScoreCrop) & (dfChoices["days_f"] < daysFCrop)]

# distribution of delays
fig1, ax1 = plt.subplots()
bins = np.arange(0, 3500, 100)
ax1.hist(dfChoices["days_f"], bins=bins, rwidth=0.8)
ax1.set_title("Bao lâu submission mới được dịch?")
ax1.set_xlabel("Số ngày")
ax1.set_ylabel("Số bài")
fig1.show()

# distribution of delays, karma and likes
fig2, ax2 = plt.subplots()
g2 = ax2.scatter(dfChoices["r_score"].values, dfChoices["days_f"].values, c=dfChoices["likes_count"].values, cmap="YlOrBr", edgecolors="None", s=30, marker="o", alpha=0.7)
ax2.set_title("Tương tác của submission trên Reddit và RedditVN")
ax2.set_xlabel("Karma trên Reddit", fontsize=16)
ax2.set_ylabel("Khoảng cách giữa bài dịch và bài gốc (ngày)")
fig2.colorbar(g2, ax=ax2)
fig2.show()

# cropped distribution of delays, karma and likes
fig3, ax3 = plt.subplots()
g3 = ax3.scatter(dfChoicesCrop["r_score"].values, dfChoicesCrop["days_f"].values, c=dfChoicesCrop["likes_count"].values, cmap="YlOrBr", edgecolors="None", s=30, marker="o", alpha=0.7)
ax3.set_title("Tương tác của submission trên Reddit và RedditVN")
ax3.set_xlabel("Karma trên Reddit")
ax3.set_ylabel("Khoảng cách giữa bài dịch và bài gốc (ngày)")
fig3.colorbar(g3, ax=ax3)
fig3.show()

# distribution of karma and likes
fig4 = plt.figure()
ax4 = fig4.add_subplot(121)
g4 = ax4.scatter(dfChoicesCrop["r_score"].values, dfChoicesCrop["likes_count"].values, alpha=0.7, marker=".")
ax4.set_title("So sánh karma và like")
ax4.set_xlabel("Karma trên Reddit")
ax4.set_ylabel("Like trên Facebook")
fig4.show()

del irr, dfChoices, delays, sampleSize, i, p, fig1, ax1, bins, fig2, ax2, g2, rScoreCrop, daysFCrop, dfChoicesCrop, fig3, ax3, g3
