def sample(**kwargs):
    result=[]
    for key ,value in kwargs.items():
        if isinstance(value,dict):
            node= [key,sample(**value)]
        elif isinstance(value,(list,tuple)):
            node =[key,[sample(_elment_=e) for e in value]]
        else:
            node=[key,value]
        result.append(node)
    return  result
