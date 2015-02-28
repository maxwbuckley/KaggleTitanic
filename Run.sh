#! /usr/bin/env bash
echo "Gogogadget scripts"
R CMD BATCH 2015TitanicScript.R
tail -3 2015TitanicScript.Rout
