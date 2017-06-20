function texttowords(text)
  #=
  Converts a string into an array of words.
  =#
  matchall(r"(\w+)",lowercase(replace(text, r"([\r\n,\.\(\)!;:\?/]|\ufeff)", s" ")))
end

function wordcount(wordvec)
  #=
  Converts an array of words into a dictionary of word counts with words as keys.
  =#
  worddict = Dict{String,Int64}()
  for w in wordvec
    worddict[w]=get(worddict,w,0) + 1
  end
  worddict
end

function adddicts(dicts)
  sumdict = Dict{String,Int64}()
  for d in dicts
    for k in keys(dicts)
      sumdict[k] = get(sumdict,k,0)+1
    end
  end
  sumdict
end

function wordfreq(worddict)
  freqdict = Dict{String,Float64}()
  totwords = sum(values(worddict))
  for k in keys(worddict)
    freqdict[k] = worddict[k]/totwords
  end
  freqdict
end

function subdicts(d1, d2)
  diffdict = d1
  for k in keys(d2)
    diffdict[k] = get(diffdict, k,0)-d2[k]
  end
  diffdict
end

function normalize(sample, control)
  #=
  Gets WordNorm_sample,control as a dict

  args: sample: a dict of the sample's word counts
        control: a dict of the control's word counts

  returns: normsample: a dict with the same keys as sample, but WordNorm_sample,control(word) as values
  =#
  normsample = Dict{String,Float64}()
  for k in keys(sample)
    c = get(control,k,0)+1
    normsample[k] = (sample[k]-c)/(sample[k]+c)
  end
  normsample
end

function compare(sample, csample)
  #=
  Gets a slightly modified version of WordNorm for comparing one sample to another. Antisymmetric in the
    sense that compare(a,b)[k] = -compare(b,a)[k].
  =#
  comp = Dict{String,Float64}()
  for k in keys(sample)
    c = get(csample,k,0)
    comp[k] = (sample[k]-c)/(sample[k]+c+1)
  end
  comp
end

function maxwords(sample, numwords)
  #=
  Gets the words with highest WordNorm in the sample.
  args: sample: WordNorm dict
        numwords: an integer
  returns: an array of pairs (word, WordNorm(word))
  =#
  sort(collect(sample), by=tuple -> last(tuple),rev=true)[1:numwords]
end
