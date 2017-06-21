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

function comparenorm(sample, csample)
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

function scoretext(textvec, scoredict, default=0)
  #=
  Gets WordNorm scores for an array of words.
  args: textvec: an array of words
        scoredict: a dict of WordNorm scores
        default: a default value for unscored words
  returns: an array of scores, 0 for words not in scoredict
  =#
  [get(scoredict, k, default) for k in textvec]
end

function colorinterp(wordscore, c1=(247,163,46), c2=(24,46,242))
  #=
  Interpolates between 1 = c1 and -1 = c2
  =#
  if wordscore > 0
    return [Int64(round(wordscore*(c1[k]-255)+255)) for k = 1:3]
  elseif wordscore == 0
    return [255,255,255]
  else
    return [Int64(round(wordscore*(255-c2[k])+255)) for k = 1:3]
  end
end
