from methods.eth_dta import eth_dta
from methods.fs_dta import fs_dta

def get_model(model_name, args):
    name = model_name.lower()
    options = {
                'eth_dta': eth_dta,
                'fs_dta': fs_dta,
               }
    return options[name](args)

