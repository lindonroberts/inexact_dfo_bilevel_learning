#!/usr/bin/env python

"""
Generic script to do runs
"""
import multiprocessing
import subprocess
import os


def have_existing_run(outfolder, setting_file, solver, accuracy):
    myfile = os.path.join(outfolder, setting_file.replace('.json', ''))
    myfile = myfile + '_%s_%s_log.txt' % (solver, accuracy)
    # print(myfile)
    if os.path.isfile(myfile):
        ifile = open(myfile, 'r')
        dfols_finished = False
        for line in ifile:
            dfols_finished = dfols_finished or line.startswith('DFO-LS finished at')
        return dfols_finished  # True if DFO-LS finished without error, otherwise False
    else:
        return False  # no output file


def run_cmd(cmd):
    # Execute the shell command 'cmd'
    subprocess.call(cmd, shell=True)
    return


def run_all_cmds(cmds_to_run, nthreads=5):
    if len(cmds_to_run) > 0:  # execute commands in parallel
        pool = multiprocessing.Pool(nthreads)
        pool.map(run_cmd, cmds_to_run)
        pool.close()
        pool.join()
    return


def get_all_setting_files(settings_folder):
    all_files = []
    for myfile in os.listdir(settings_folder):
        if myfile.endswith('.json'):
            all_files.append(myfile)
    return all_files


def main():
    EXEC_FILE = 'dfols_wrapper.py'

    ALL_SOLVERS = []
    ALL_SOLVERS.append(('fista', 'dynamic'))
    ALL_SOLVERS.append(('fista', 'fixed_niters_200'))
    ALL_SOLVERS.append(('fista', 'fixed_niters_2000'))
    ALL_SOLVERS.append(('gd', 'dynamic'))
    ALL_SOLVERS.append(('gd', 'fixed_niters_1000'))
    ALL_SOLVERS.append(('gd', 'fixed_niters_10000'))

    SETTINGS_FOLDER = 'problem_settings'
    OUTFOLDER = os.path.join(os.pardir, 'raw_results')
    nthreads = 6

    skip_existing_files = True  # clean runs from scratch
    # skip_existing_files = False  # redoing failed runs

    # print_runs = True  # display list of runs
    print_runs = False  # do runs

    setting_files = get_all_setting_files(SETTINGS_FOLDER)

    cmds_to_run = []
    num_cmds = 0
    for setting_file in sorted(setting_files):
        if not setting_file.endswith('.json'):
            setting_file += '.json'
        this_outfolder = os.path.join(OUTFOLDER, setting_file.replace('.json', ''))
        for (solver, accuracy) in ALL_SOLVERS:
            cmd = "python %s %s %s %s %s" % \
                  (EXEC_FILE, os.path.join(SETTINGS_FOLDER, setting_file), solver, accuracy, OUTFOLDER)
            run_already_done = have_existing_run(this_outfolder, setting_file, solver, accuracy)
            if print_runs:
                if (not skip_existing_files) or not run_already_done:
                    num_cmds += 1
                    print(cmd)
                # else:
                #     print("Skipping %s %s %s" % (setting_file, solver, accuracy))
            else:
                if skip_existing_files and run_already_done:
                    # print("Skipping %s %s %s" % (setting_file, solver, accuracy))
                    continue  # skip
                cmds_to_run.append(cmd)

    if not print_runs:
        if not os.path.isdir(OUTFOLDER):
            os.makedirs(OUTFOLDER, exist_ok=True)
        run_all_cmds(cmds_to_run, nthreads=nthreads)
    else:
        print("Need to do %g runs" % num_cmds)
    print('Done')
    return


if __name__ == '__main__':
    main()
