def preparse_loras_multipliers(loras_multipliers):
    if isinstance(loras_multipliers, list):
        return [multi.strip(" \r\n") if isinstance(multi, str) else multi for multi in loras_multipliers]

    loras_multipliers = loras_multipliers.strip(" \r\n")
    loras_mult_choices_list = loras_multipliers.replace("\r", "").split("\n")
    loras_mult_choices_list = [multi.strip() for multi in loras_mult_choices_list if len(multi)>0 and not multi.startswith("#")]
    loras_multipliers = " ".join(loras_mult_choices_list)
    return loras_multipliers.split(" ")

def expand_slist(slists_dict, mult_no, num_inference_steps, model_switch_step ):
    def expand_one(slist, num_inference_steps):
        if not isinstance(slist, list): slist = [slist]
        new_slist= []
        if num_inference_steps <=0:
            return new_slist
        inc =  len(slist) / num_inference_steps 
        pos = 0
        for i in range(num_inference_steps):
            new_slist.append(slist[ int(pos)])
            pos += inc
        return new_slist

    phase1 = slists_dict["phase1"][mult_no]
    phase2 = slists_dict["phase2"][mult_no]
    if isinstance(phase1, float) and isinstance(phase2, float) and phase1 == phase2:
        return phase1 
    return expand_one(phase1, model_switch_step) + expand_one(phase2, num_inference_steps - model_switch_step)

def parse_loras_multipliers(loras_multipliers, nb_loras, num_inference_steps, merge_slist = None, max_phases = 2, model_switch_step = None):
    if model_switch_step is None:
        model_switch_step = num_inference_steps
    def is_float(element: any) -> bool:
        if element is None: 
            return False
        try:
            float(element)
            return True
        except ValueError:
            return False
    loras_list_mult_choices_nums = []
    slists_dict = { "model_switch_step": model_switch_step}
    slists_dict["phase1"] = phase1 = [1.] * nb_loras
    slists_dict["phase2"] = phase2 = [1.] * nb_loras

    if isinstance(loras_multipliers, list) or len(loras_multipliers) > 0:
        list_mult_choices_list = preparse_loras_multipliers(loras_multipliers)
        for i, mult in enumerate(list_mult_choices_list):
            current_phase = phase1
            if isinstance(mult, str):
                mult = mult.strip()
                phase_mult = mult.split(";")
                shared_phases = len(phase_mult) <=1
                if len(phase_mult) > max_phases:
                    return "", "", f"Loras can not be defined for more than {max_phases} Denoising phase{'s' if max_phases>1 else ''} for this model"
                for phase_no, mult in enumerate(phase_mult):
                    if phase_no > 0: current_phase = phase2
                    if "," in mult:
                        multlist = mult.split(",")
                        slist = []
                        for smult in multlist:
                            if not is_float(smult):                
                                return "", "", f"Lora sub value no {i+1} ({smult}) in Multiplier definition '{multlist}' is invalid"
                            slist.append(float(smult))
                    else:
                        if not is_float(mult):                
                            return "", "", f"Lora Multiplier no {i+1} ({mult}) is invalid"
                        slist = float(mult)
                    if shared_phases:
                        phase1[i] = phase2[i] = slist
                    else:
                        current_phase[i] = slist
            else:
                phase1[i] = phase2[i] = float(mult)

    if merge_slist is not None:
        slists_dict["phase1"] = phase1 = merge_slist["phase1"] + phase1
        slists_dict["phase2"] = phase2 = merge_slist["phase2"] + phase2

    loras_list_mult_choices_nums = [ expand_slist(slists_dict, i, num_inference_steps, model_switch_step )  for i in range(len(phase1)) ]
    loras_list_mult_choices_nums = [ slist[0] if isinstance(slist, list) else slist for slist in loras_list_mult_choices_nums ]
    
    return  loras_list_mult_choices_nums, slists_dict, ""

def update_loras_slists(trans, slists_dict, num_inference_steps, model_switch_step = None ):
    from mmgp import offload
    sz = len(slists_dict["phase1"])
    slists = [ expand_slist(slists_dict, i, num_inference_steps, model_switch_step ) for i in range(sz)  ]
    nos = [str(l) for l in range(sz)]
    offload.activate_loras(trans, nos, slists ) 

