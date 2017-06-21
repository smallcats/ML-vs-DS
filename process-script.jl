#script for getting WordNorm for Machine Learning and Data

include("loadDesc.jl")
include("process-text.jl")

desc = loadDesc()

controlwords = wordcount(texttowords(join(desc[1]," ")));
datawords = wordcount(texttowords(join(desc[2]," ")));
mlwords = wordcount(texttowords(join(desc[3]," ")));

mlnorm = normalize(mlwords,controlwords);
datanorm = normalize(datawords,controlwords);

mlvsds = comparenorm(mlwords, datawords);

insighttextvec = texttowords(loadInsight());
insightscores = scoretext(insighttextvec, mlvsds);
