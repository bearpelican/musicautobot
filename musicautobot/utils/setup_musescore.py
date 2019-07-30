def setup_musescore(musescore_path=None):
    if not is_ipython(): return
    
    import platform
    from music21 import environment
    from pathlib import Path
    
    system = platform.system()
    if system == 'Linux':
        import os
        os.environ['QT_QPA_PLATFORM']='offscreen' # https://musescore.org/en/node/29041
        
    existing_path = environment.get('musicxmlPath')
    if existing_path: return
    if musescore_path is None:
        if system == 'Darwin':
            app_paths = list(Path('/Applications').glob('MuseScore *.app'))
            if len(musescore_app): musescore_path = app_paths[-1]/'Contents/MacOS/mscore'
        elif system == 'Linux':
            musescore_path = '/usr/bin/musescore'
    
    if musescore_path is None or not Path(musescore_path).exists():
        print('Warning: Could not find musescore installation. Please install musescore (see README) and/or update music21 environment paths')
    else :
        environment.set('musicxmlPath', musescore_path)
        environment.set('musescoreDirectPNGPath', musescore_path)

def is_ipython():
    try: get_ipython
    except: return False
    return True