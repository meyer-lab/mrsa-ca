#!/bin/bash

echo "How many threads do you want to use?"
read THREADS

#start conda environment
conda init --all
conda activate rnaseq_salmon

#check to make sure the salmon_gene_counts directory exists (setup_salmon.sh should have created it)
if [ ! -d "./salmon_processing" ]; then
    echo "salmon_processing directory not found. Please run setup_salmon.sh first"
    exit 1
fi

#ask for single or paired end reads
echo "Is this [single] end or [paired] end data?"
read endType
if [ "$endType" = "single" ]; then
    echo "selected [single]"
elif [ "$endType" = "paired" ]; then
    echo "selected [paired]"
else
    echo "input either [single] or [paired]"
    exit 1
fi

#ask for the number of SRA accessions to process at a time
echo "How many SRA accessions would you like to process at time?"
read batchSize

#navigate to sra_out to begin processing the SRA accessions
cd ./salmon_processing/sra_out

#check if the accession list exists
if [ ! -f "accession_list.txt" ]; then
    echo "accession_list.txt not found. Please run setup_salmon.sh first"
    exit 1
fi

#remove empty lines from the accession_list.txt file
sed -i '/^$/d' accession_list.txt

#break the accession list into batches and process them indicidually
#count the number of lines in the given accession list
lineCount=$(wc -l < accession_list.txt)
#calculate the number of batches to process
batchCount=$(($lineCount / $batchSize))
#add one to the batch count if there are any remaining accessions
if [ $(($lineCount % $batchSize)) -ne 0 ]; then
    batchCount=$(($batchCount + 1))
fi

#loop through the accession list in batches, and perform salmon quantification
for i in $(seq 1 $batchCount); do
    #calculate the start and end lines for the batch
    startLine=$((($i - 1) * $batchSize + 1))
    endLine=$(($i * $batchSize))
    #check if the end line is greater than the total number of lines to avoid out of bounds
    if [ $endLine -gt $lineCount ]; then
        endLine=$lineCount
    fi

    #extract the batch of accessions
    sed -n "${startLine},${endLine}p" accession_list.txt > batch_accessions.txt
    #If text file is generated through gui or Windows, remove \r after each line. Maybe not needed with sed making separate batch_accession.txt?
    sed -i 's/\r$//' batch_accessions.txt

    # Remove empty lines from the batch_accessions.txt file
    sed -i '/^$/d' batch_accessions.txt
    
    # Loop through the batch of accessions, fetch samples and fastq format them
    while IFS="" read -r line || [ -n "$line" ]; do
        echo "Processing $line"
        if [ "$endType" = "single" ]; then
            fasterq-dump $line --progress --threads $THREADS 
        elif [ "$endType" = "paired" ]; then
            fasterq-dump $line --split-files --skip-technical --progress --threads $THREADS
        fi
    done < batch_accessions.txt

    #gzip all fastq files generated
    echo "Gzipping all fastq files"
    gzip --verbose *.fastq

    #navigate back to the main directory
    cd ..

    #quantify expression using salmon
    #loop through the fastq.gz files and quantify expression, checking for single or paired end reads
    echo "Quantifying expression with Salmon"
    if [ "$endType" = "single" ]; then
        for file in ./sra_out/*.fastq.gz; do
            echo "Processing $file"
            salmon quant -p $THREADS -i ./salmon_ref/salmon_index --geneMap ./salmon_ref/mappings.gtf --validateMappings --gcBias -l A -r ./sra_out/"$file" -o ./salmon_gene_counts/"$(basename "$file" .fastq.gz)"
        done
    elif [ "$endType" = "paired" ]; then
        for file in ./sra_out/*_1.fastq.gz; do
            echo "Processing $file"
            base=$(basename "$file" _1.fastq.gz)
            salmon quant -p $THREADS -i ./salmon_ref/salmon_index --geneMap ./salmon_ref/mappings.gtf --validateMappings --gcBias -l A -1 ./sra_out/"${base}_1.fastq.gz" -2 ./sra_out/"${base}_2.fastq.gz" -o ./salmon_gene_counts/"${base}"
        done
    fi

    #clean up the fastq files to save space and prepare for the next batch
    #navigate back to sra_out to remove fastq files and start the next batch
    cd ./sra_out
    #remove all fastq files
    rm *.fastq.gz
    #remove the batch_accessions.txt file
    rm batch_accessions.txt

done

#navigate back to the main directory
cd ..

# aggregate all the gene counts into a single file
echo "Aggregating gene counts..."

# create simplified gene count files for each sample
while IFS="" read -r sample || [ -n "$sample" ]; do
    tail -n +2 ./salmon_gene_counts/${sample}/quant.genes.sf | cut -f 4 > ./salmon_gene_counts/${sample}.count
done < ./sra_out/accession_list.txt

# Set ${sample} to the first line of accession_list.txt
sample=$(head -n 1 ./sra_out/accession_list.txt)
# create a list of all the genes
tail -n +2 ./salmon_gene_counts/${sample}/quant.genes.sf | cut -f 1 > ./salmon_gene_counts/genes.txt

# combine all the gene counts into a single file, with gene names as the first column
paste ./salmon_gene_counts/genes.txt ./salmon_gene_counts/*.count > ./salmon_gene_counts/all_counts.txt
# create a header for the file with sorted sample names (since we pasted them with a wildcard)
sed -i "1i gene\t$(sort ./sra_out/accession_list.txt | tr '\n' '\t')" ./salmon_gene_counts/all_counts.txt

#clean up the gene count files
rm ./salmon_gene_counts/*.count

echo "All SRA accessions processed and expression quantified with Salmon"
echo "Results can be found in the salmon_gene_counts directory"