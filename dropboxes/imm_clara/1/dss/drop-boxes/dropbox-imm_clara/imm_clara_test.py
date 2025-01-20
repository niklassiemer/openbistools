def process(transaction):
  dataSet = transaction.createNewDataSet()
  exp = transaction.getExperiment("/DEFAULT/DEFAULT/DEFAULT")
  dataSet.setExperiment(exp)
  transaction.moveFile(transaction.getIncoming().getAbsolutePath(), dataSet)
