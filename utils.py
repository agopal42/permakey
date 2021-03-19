ENV_NAME_TO_GYM_NAME = {
    "seaquest": "SeaquestNoFrameskip-v4",
    "frostbite": "FrostbiteNoFrameskip-v4",
    "space_invaders": "SpaceInvadersNoFrameskip-v4",
    "mspacman": "MsPacmanNoFrameskip-v4",
    "battlezone": "BattleZoneNoFrameskip-v4",
    "enduro": "EnduroNoFrameskip-v4",
    "montezuma_revenge": "MontezumaRevengeNoFrameskip-v4"}


def add_sacred_log(key, value, _run):
    """
    Adds logs to the Sacred run info dict. Creates new dicts along the (dotted)
    key path if not available, and appends the value to a list at the final
    destination.
    :param key: (dotted) path.to.log.location.
    :param value: (scalar) value to append at log location
    :param _run: _run dictionary of the current experiment
    """

    if 'logs' not in _run.info:
        _run.info['logs'] = {}
    logs = _run.info['logs']
    split_path = key.split('.')
    current = logs
    for p in split_path[:-1]:
        if p not in current:
            current[p] = {}
        current = current[p]

    final_key = split_path[-1]
    if final_key not in current:
        current[final_key] = []
    entries = current[final_key]
    entries.append(value)