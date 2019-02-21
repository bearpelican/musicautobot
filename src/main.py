import os
import math
import json
import time
import multiprocessing as mp
from tab_parser import proc_xml
import matplotlib.pyplot as plt
from roman_to_symbol import proc_roman_to_symbol
from to_pianoroll import proc_event_to_midi, proc_midi_to_pianoroll
plt.switch_backend('agg')

root_dir = '../datasets'
root_xml = '../datasets/xml'
root_event = '../datasets/event'
root_pianoroll = '../datasets/pianoroll'
log_dir = '../analysis/log'


def traverse_dir(root_dir, extension='.xml', is_pure=True):
    print('[*] Scanning...')
    file_list = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension):
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir)+1:] if is_pure else mix_path
                file_list.append(pure_path)

    return file_list


def error_handler(e, url, log, side_info=''):
    print(e)
    print(url, '\n')
    log.write(side_info + '  -  '+str(e) + '\n' + url + '\n')


def split_file(path):
    base = os.path.basename(path)
    filename = base.split('.')[0]
    extension = base.split('.')[1]
    return path[:-len(base)-1], filename, extension


def proc(xml_list, index=0):
    num_xml = len(xml_list)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log = open(os.path.join(log_dir, 'log_'+str(index)+'.txt'), 'w')

    cnt_ok = 1
    cnt_total = 0
    for xidx in range(num_xml):  # num_xml
        cnt_total += 1
        path, fn, _ = split_file(xml_list[xidx])
        tmp = path.split('/')
        url = 'https://www.hooktheory.com/theorytab/view/' + tmp[-2] + '/' + tmp[-1]
        try:
            current = '[%d] (%d/%d)[%d/%d] %s' % (index, xidx, num_xml, cnt_ok, cnt_total, xml_list[xidx])
            print(current)

            # path arrangement
            path_xml = os.path.join(root_xml, path)
            path_event = os.path.join(root_event, path)
            path_pianoroll = os.path.join(root_pianoroll, path)

            # roman
            xml_file = os.path.join(path_xml, fn+'.xml')
            name = fn + '_roman'

            try:
                raw_roman = proc_xml(xml_file, save_path=path_event, name=name, save_type='json')
            except Exception as e:
                print('> Broken File!!!')
                error_handler(e, url, log, current)
                continue

            # to event symbol
            name = fn + '_symbol_key'
            raw_symbol_key = proc_roman_to_symbol(
                                    raw_roman,
                                    save_path=path_event,
                                    name=name,
                                    save_type='json',
                                    is_key=True)

            name = fn + '_symbol_nokey'
            raw_symbol_nokey = proc_roman_to_symbol(
                                    raw_roman,
                                    save_path=path_event,
                                    name=name,
                                    save_type='json',
                                    is_key=False)

            # to midi
            proc_event_to_midi(raw_symbol_key, save_path=path_pianoroll, name=fn+'_key')
            midi = proc_event_to_midi(raw_symbol_nokey, save_path=path_pianoroll, name=fn+'_nokey')

            # to pianoroll

            beats_in_measure = int(raw_symbol_nokey['metadata']['beats_in_measure'])
            pianoroll = proc_midi_to_pianoroll(midi, beats_in_measure)
            pianoroll.save(os.path.join(path_pianoroll, fn+'.npz'))

            # plot
            pianoroll.plot()
            plt.savefig(os.path.join(path_pianoroll, fn+'.png'), dpi=500)
            plt.close()

            cnt_ok += 1
        except Exception as e:
            error_handler(e, url, log, current)

    print('Done!!!')
    log.close()
    queue.put((cnt_ok, cnt_total))


if __name__ == '__main__':

    xml_list = traverse_dir(root_xml)

    with open(os.path.join(root_dir, 'xml_list.json'), "w") as f:
        json.dump(xml_list, f)

    with open(os.path.join(root_dir, 'xml_list.json'), "r") as f:
        xml_list = json.load(f)

    # dynamic multi-process
    queue = mp.Queue()
    processes = []
    amount = len(xml_list)
    # n_cpu = mp.cpu_count()  # number of core
    n_cpu = 3
    amount_batch = math.ceil(amount/n_cpu)
    print('cpu count: %d, batch size: %d, total: %d' % (n_cpu, amount_batch, amount))

    for process_idx in range(n_cpu):
        st = process_idx * amount_batch
        ed = min(st+amount_batch, amount)
        print('ps - %d [from %d to %d]' % (process_idx, st, ed))
        processes.append(mp.Process(target=proc, args=(xml_list[st:ed], process_idx)))
        processes[-1].start()

    time.sleep(5)

    start_time = time.time()
    results = []
    for i in range(n_cpu):
        results.append(queue.get())

    for process in processes:
        process.join()

    total_ok = 0
    total_proc = 0
    for r in results:
        total_ok += r[0] - 1
        total_proc += r[1]

    print('OK: %d in %d' % (total_ok, total_proc))
    print("Elapsed time: %s" % (time.time() - start_time))
