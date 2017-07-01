#script for getting WordNorm for Machine Learning and Data

include("loadDesc.jl");
include("process-text.jl");

desc = loadDesc()
#insighttext = loadInsight();
desc = [join(desc[k], " ") for k=1:3];

#controlwords = wordcount(texttowords(desc[1]));
datawords = wordcount(texttowords(desc[2]));
mlwords = wordcount(texttowords(desc[3]));

#mlnorm = normalize(mlwords,controlwords);
#datanorm = normalize(datawords,controlwords);

mlvsds = comparenorm(mlwords, datawords);

#insighttextvec = texttowords(insighttext);
#insightscore = totaltextscore(insighttextvec, mlvsds);
conttextvec = texttowords(desc[1]);
contscore = totaltextscore(conttextvec, mlvsds);
