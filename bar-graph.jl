#Script for graphing ml vs. ds words

#arrays of pairs word => WordNorm_ds/ml(word) for top 10 ds/ml words in order
dstop = maxwords(datanorm, 10)
mltop = maxwords(mlnorm, 10)

#arrays of WordNorm_ds/ml(word) for top 10 ds/ml words in order
dstopnum = [dstop[k][2] for k=1:10]
mltopnum = [mltop[k][2] for k=1:10]

#arrays of word for top 10 ds/ml words in order
dstopword = [dstop[k][1] for k=1:10]
mltopword = [mltop[k][1] for k=1:10]

#arrays of WordNorm_ml/ds(word) for top 10 ds/ml words in order
dstopmlnum = [get(mlnorm,k,-1) for k in dstopword]
mltopdsnum = [get(datanorm,k,-1) for k in mltopword]

m="Machine Learning"
d="Data Science"

bar([1:10 1:10 0.85:9.85 0.85:9.85], [dstopmlnum mltopdsnum dstopnum mltopnum], 
			label=[nothing "$d WordNorm" nothing "$m WordNorm"], legend=[false true], 
			bar_width = 0.8, layout = 2, ylims = (-1,1), 
			title=["WordNorm for Top 10 $d Words" "WordNorm for Top 10 $m Words"],
			left_margin = 2*mm, right_margin=2*mm, top_margin=2*mm, bottom_margin=5*mm,
			xrotation = rad2deg(pi/3), size = (1000,500),
			xticks = [(1:10, dstopword) (1:10, mltopword)],
			color = [:orange :blue :blue :orange])
