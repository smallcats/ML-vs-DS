#Script for graphing ml vs. ds words

function barWordNormCtrl(datanorm, mlnorm)
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
end

function produceLaTeXcolored!(text, contav, scoredict, numwords)
	#=
	creates a string of LaTeX code to produce the text given with the approximate
		score shown.
	relevant LaTeX prewritten code to make the LaTeX code compile:
	\usepackage{xcolor}
	\definecolor{o5}{RGB}{247,163,46}
	\definecolor{o4}{RGB}{249,181,88}
	\definecolor{o3}{RGB}{250,200,130}
	\definecolor{o2}{RGB}{252,218,171}
	\definecolor{o1}{RGB}{253,237,213}
	\definecolor{w}{RGB}{255,255,255}
	\definecolor{b1}{RGB}{209,213,252}
	\definecolor{b2}{RGB}{163,171,250}
	\definecolor{b3}{RGB}{116,130,247}
	\definecolor{b4}{RGB}{70,88,245}
	\definecolor{b5}{RGB}{242,46,242}
	=#
	words = texttowords(text)[1:numwords]
	textscores = scoretextwords(words, scoredict)
	normedtextscores = [tanh(atanh(k)-atanh(contav)) for k in textscores]
	wordcodes = []
	while length(words) > 0
		word = shift!(words)
		score = shift!(normedtextscores)
		score = Int8(round(score*5))
		if score > 0
			color = "o$(score)"
		elseif score < 0
			color = "b$(-score)"
		else
			color = "w"
		end
		wordcode = "{\\color{$(color)}$(word)}"
		push!(wordcodes, wordcode)
	end
	join(wordcodes," ")
end
