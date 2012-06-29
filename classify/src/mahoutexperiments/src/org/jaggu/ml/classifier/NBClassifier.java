/**
 * Mahout NaiveBayes Classifier Example
 */
package org.jaggu.ml.classifier;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;

import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.util.Version;
import org.apache.mahout.classifier.ClassifierResult;
import org.apache.mahout.classifier.bayes.Algorithm;
import org.apache.mahout.classifier.bayes.BayesAlgorithm;
import org.apache.mahout.classifier.bayes.BayesParameters;
import org.apache.mahout.classifier.bayes.ClassifierContext;
import org.apache.mahout.classifier.bayes.Datastore;
import org.apache.mahout.classifier.bayes.InMemoryBayesDatastore;
import org.apache.mahout.classifier.bayes.InvalidDatastoreException;

/**
 * @author Jaganadh G
 * 
 */
public class NBClassifier {

	/**
	 * @param
	 * @retrun bayesParams
	 */

	public static BayesParameters setParams() {
		BayesParameters bayesParams = new BayesParameters();
		bayesParams.setGramSize(3);
		bayesParams.set("dataSource", "hdfs");
		bayesParams.set("defaultCat", "unknown");
		bayesParams.set("encoding", "UTF-8");
		bayesParams.set("alpha_i", "1.0");

		return bayesParams;
	}

	/**
	 * @param strModelPath
	 * @throws InvalidDatastoreException
	 * @return classifier
	 * @throws IOException
	 */

	public static ClassifierContext initClassifier(File strModelPath)
			throws InvalidDatastoreException, IOException {
		BayesParameters params = setParams();
		params.setBasePath(strModelPath.getCanonicalPath());
		Datastore ds = new InMemoryBayesDatastore(params);
		Algorithm algo = new BayesAlgorithm();
		ClassifierContext classifier = new ClassifierContext(algo, ds);
		classifier.initialize();
		return classifier;

	}

	/**
	 * @param args
	 * @return preparedDoc
	 * @throws IOException
	 */

	public static String[] preprocessDocument(File fnDocument)
			throws IOException {
		StandardAnalyzer analyzer = new StandardAnalyzer(Version.LUCENE_31);
		TokenStream tokensteam = analyzer
				.tokenStream(null, new InputStreamReader(new FileInputStream(
						fnDocument), "UTF-8"));
		ArrayList<String> tokenlist = new ArrayList<String>();

		while (tokensteam.incrementToken()) {
			tokenlist.add(tokensteam.getAttribute(CharTermAttribute.class)
					.toString());

		}
		String[] preparedDoc = tokenlist.toArray(new String[tokenlist.size()]);

		return preparedDoc;

	}

	/**
	 * @param args
	 * @throws IOException
	 * @throws InvalidDatastoreException
	 */
	public static void main(String[] args) throws InvalidDatastoreException,
			IOException {
		File model_path = new File(args[0]);
		File input_dir = new File(args[1]);

		ClassifierContext classifier = initClassifier(model_path);
		String[] list_files = input_dir.list();

		for (int i = 0; i < list_files.length; i++) {
			if (list_files[i].toString().endsWith(".txt")) {
				String[] doc = preprocessDocument(new File(list_files[i]));
				ClassifierResult[] classifier_result = classifier
						.classifyDocument(doc, "unknown", 3);

				for (ClassifierResult result : classifier_result) {
					System.out.println("Category: " + result.getLabel()
							+ " Score: " + result.getScore());
				}
			}
		}

	}

}
