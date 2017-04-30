function processtext(text)
  matchall(r"(\w+)",lowercase(replace(text, r"([\r\n,\.\(\)!;:\?/])", s"")))
end
