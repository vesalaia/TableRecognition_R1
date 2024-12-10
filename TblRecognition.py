#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import sys
sys.path.append('/users/vesalaia/pipelinv2')
sys.path.append("/users/vesalaia/.local/lib/python3.9/site-packages/bin")
sys.path.append("/users/vesalaia/.local/lib/python3.9/site-packages/lib/python3.9/site-packages")

import logging
import datetime
import time
start_time = time.time()


from config.options import Options

from data.dataset import OCRDatasetInstanceSeg
from pipeline.engine import initFolder, extractText, pipelineTask
from utils.config_check import configuration_ok

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
#
# Main progranm
#

def setup_logger(log_file_name):
    logging.basicConfig(
        filename=log_file_name,
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

def main(start_time):
    parser = argparse.ArgumentParser(description='Utilities for E2E Text recognition  pipeline')

    parser.add_argument('--task', dest='executeTask', type=str, 
                        help='''Describes the task to be performed, options:\n
                        init: Creates PAGE XML file for each image\n 
                        region: Region detection for page images\n 
                        line:: Line detection for page images\n 
                        table: Table structure detection for page images\n
                        recognize: Text recognition \n
                        update: Updates the PAGE XML \n
                        text: Export document structure and text into a text file\n
                        json: Export document structure and text into a json file\n
                        csv: Export table structure and text into a csv file\n''')
    parser.add_argument('--log', dest='logfile', type=str, 
                        help='Location of log file')
    parser.add_argument('--dir', dest='infolder', type=str, 
                        help='Directory of images (.jpg or .jpeg)')
    parser.add_argument('--inpage', dest='inpage', type=str, 
                        help='Input directory for PAGE XML files')
    parser.add_argument('--outpage', dest='outpage', type=str, 
                        help='Output directory for PAGE XML files')
    parser.add_argument('--config', dest='cfgfile', type=str, 
                        help='Location of configuration file')
    parser.add_argument('--linedir', dest='linedir', type=str, 
                        help='Input directory of Line images')
    parser.add_argument('--linpage', dest='linpage', type=str, 
                        help='Input directory of Line image PAGE XML files')
    parser.add_argument('--regiondir', dest='regiondir', type=str, 
                        help='Input directory of region images')
    parser.add_argument('--rinpage', dest='rinpage', type=str, 
                        help='Input directory of Region image PAGE XML files')
    parser.add_argument('--merge',  dest='merge', type=str, help="Try overlapping region merge")
    parser.add_argument('--ro',  dest='ro', type=str, help="Check the reading order of regions")
    parser.add_argument('--combine', dest='combine', type=str, help="Try combining lines")

  #  parser.add_argument('--merge', dest='tryMerge',type=str2bool, nargs='?', default=False, 
  #                      help='Try overlapping region merge')
  #  parser.add_argument('--combine', dest='combine', type=str2bool, nargs='?', default=False, 
  #                      help='Combine small lines if possible')
  #  parser.add_argument('--ro', dest='reading_order', type=str2bool, nargs='?', default=False, 
  #                      help='Check the reading order of regions')
    args = parser.parse_args()
    logfile = args.logfile
    if args.logfile:
        log_file_name = f"{os.path.basename(logfile)}"
    else:
        log_file_name = f"table_recognition.log"

    setup_logger(log_file_name)

    cfgfile = args.cfgfile
    if os.path.exists(cfgfile):
        opts =  Options(cfgfile)
    else:
        logging.error(f"Configuration file: {cfgfile} does not exist")
        opts = None

    logging.info(f"Configuration file: {cfgfile}")

    executeTask = args.executeTask
    infolder = args.infolder
    inpage = args.inpage
    outpage = args.outpage
    linedir = args.linedir
    linpage = args.linpage
    regiondir = args.regiondir
    rinpage = args.rinpage
    if args.merge:
        tryMerge = str2bool(args.merge)
    else:
        tryMerge = False
    if args.combine:
        combine = str2bool(args.combine)
    else:
         combine = False
    if args.ro:
        reading_order = str2bool(args.ro)
    else:
        reading_order = False


    if opts != None:
        if configuration_ok(opts, executeTask.lower()):    
            if executeTask.lower() in ["init", "i"]:
                executeTask = "init"
                if outpage == None:
                    outpage = "page"
                if infolder != None:
                    logging.info(f"Task:{executeTask} Folder:{infolder} XML-output: {outpage}")
                    initFolder(opts, infolder, outpage) 
            elif executeTask.lower() in ["text", "json", "csv"]:
                executeTask = executeTask.lower()
                if inpage == None:
                    inpage = "pageText"
                if outpage == None:
                    outpage = "text"
                if infolder != None:
                    logging.info(f"Task:{executeTask} Folder:{infolder} XML-input:{inpage} XML-output: {outpage}")
                    extractText(opts, infolder, inpage, outpage)
            elif executeTask.lower() in ["detectregion", "dr", "region"]:
                executeTask = "region"
                if inpage == None:
                    inpage = "page"
                if outpage == None:
                    outpage = "pageRD"
                if infolder != None:
                    RO_groups = opts.RO_region_groups
                    logging.info(f"Task:{executeTask} Folder:{infolder} XML-input:{inpage} XML-output: {outpage}")
                    pipelineTask(opts, executeTask, infolder, inpage=inpage, outpage=outpage, tryMerge=tryMerge, 
                                 reading_order=reading_order)
            elif executeTask.lower() in ["detectlines", "dl", "line"]:
                executeTask = "line"
                if inpage == None:
                    inpage = "pageRD"
                if outpage == None:
                    outpage = "pageLD"
                if infolder != None:
                    RO_groups = opts.RO_region_groups
                    one_page_per_image = True
                    logging.info(f"Task:{executeTask} Folder:{infolder} XML-input:{inpage} XML-output: {outpage}")
                    pipelineTask(opts, executeTask, infolder, inpage=inpage, outpage=outpage, tryMerge=tryMerge,
                                 reading_order=reading_order, line_model="mask r-cnn")
            elif executeTask.lower() in ["recognizetext", "rt", "recognize"]:
                executeTask = "recognize"
                if inpage == None:
                    inpage = "pageLD"
                if outpage == None:
                    outpage = "pageText"
                if infolder != None:
                    RO_groups = opts.RO_region_groups
                    logging.info(f"Task:{executeTask} Folder:{infolder} XML-input:{inpage} XML-output: {outpage}")
                    pipelineTask(opts, executeTask, infolder, inpage=inpage, outpage=outpage, 
                                 reading_order=reading_order)
            elif executeTask.lower() in ["update", "u"]:
                executeTask = "update"
                if inpage == None:
                    inpage = "pageLD"
                if outpage == None:
                    outpage = "pageU"
                    if infolder != None:
                        RO_groups = opts.RO_line_groups
                        logging.info(f"Task:{executeTask} Folder:{infolder} XML-input:{inpage} XML-output: {outpage}")
                        pipelineTask(opts, executeTask, infolder, inpage=inpage, outpage=outpage, 
                                     reading_order=reading_order, combine=combine)
            elif executeTask.lower() in ["table", "t"]:
                executeTask = "table"
                if inpage == None:
                    inpage = "page"
                if outpage == None:
                    outpage = "pageTbl"
                if infolder != None:
                    logging.info(f"Task:{executeTask} Folder:{infolder} XML-input:{inpage} XML-output: {outpage}")
                    pipelineTask(opts, executeTask, infolder, inpage=inpage, outpage=outpage, 
                                 reading_order=False, combine=False)
            elif executeTask.lower() in ["cell", "c"]:
                executeTask = "cell"
                if inpage == None:
                    inpage = "pageTbl"
                if outpage == None:
                    outpage = "pageCell"
                if infolder != None:
                   logging.info(f"Task:{executeTask} Folder:{infolder} XML-input:{inpage} XML-output: {outpage}")
                   pipelineTask(opts, executeTask, infolder, inpage=inpage, outpage=outpage, 
                                reading_order=False, combine=False)
            elif executeTask.lower() in ["cellrecognize", "cr"]:
                executeTask = "cellrecognize"
                if inpage == None:
                    inpage = "pageCell"
                if outpage == None:
                    outpage = "pageText"
                if infolder != None:
                    logging.info(f"Task:{executeTask} Folder:{infolder} XML-input:{inpage} XML-output: {outpage}")
                    pipelineTask(opts, executeTask, infolder, inpage=inpage, outpage=outpage)
            else:
                logging.error(f"Task not recognized: {executeTask}")
    # print some statistics
        elapsed_time = time.time() - start_time
        logging.info(f"Statistics: Program execution time: {elapsed_time:.2f} seconds, {elapsed_time/3600:.2f} hours")

        if opts.statistics['files_processed'] != 0:
            logging.info(f"Statistics: Files processed: {opts.statistics['files_processed']}, {elapsed_time/opts.statistics['files_processed']:.2f} seconds per file")

        if opts.statistics['regions_detected'] != 0:
            logging.info(f"Statistics: Regions detected: {opts.statistics['regions_detected']}")
        
        if opts.statistics['tables_detected'] != 0:
            logging.info(f"Statistics: Tables detected: {opts.statistics['tables_detected']}")
    
        if opts.statistics['columns_detected'] != 0:
            logging.info(f"Statistics: Columns detected: {opts.statistics['columns_detected']}")
    
        if opts.statistics['rows_detected'] != 0:
            logging.info(f"Statistics: Rows detected: {opts.statistics['rows_detected']}")
    
        if opts.statistics['cells_detected'] != 0:
            logging.info(f"Statistics: Cells detected: {opts.statistics['cells_detected']}")
    
        if opts.statistics['lines_detected'] != 0:
            logging.info(f"Statistics: Lines detected: {opts.statistics['lines_detected']}")
    
        if opts.statistics['lines_recognized'] != 0:
            logging.info(f"Statistics: Lines recognized: {opts.statistics['lines_recognized']}")
    
        if opts.statistics['cell_count'] != 0:
           logging.info(f"Statistics: Total number of cells processed: {opts.statistics['cell_count']}")

if __name__ == "__main__":
    main(start_time)
