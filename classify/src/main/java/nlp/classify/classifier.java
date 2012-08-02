package nlp.classify;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Properties;
import java.util.Scanner;

import org.apache.commons.io.FileUtils;
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
 * @author Sudharshan
 * 
 */
public class classifier {

	private static Properties props = new Properties();
	private static BayesParameters params = new BayesParameters();
	public static Scanner in = new Scanner(System.in);

	public static ClassifierContext setParams(File strModelPath) {
		params.setGramSize(2);
		params.set("dataSource", "hdfs");
		params.set("defaultCat", "unknown");
		params.set("encoding", "UTF-8");
		params.set("alpha_i", "1.0");
		try {
			params.setBasePath(strModelPath.getCanonicalPath());
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		Datastore ds = new InMemoryBayesDatastore(params);
		Algorithm algo = new BayesAlgorithm();
		ClassifierContext predict = new ClassifierContext(algo, ds);
		try {
			predict.initialize();
		} catch (InvalidDatastoreException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return predict;
	}

	/**
	 * @param args
	 * @return preparedDoc
	 * @throws IOException
	 */
	public static void testClassifier(String modelPath, String inputDirPath) {
		File model = new File(modelPath);
		File[] inputDir = new File(inputDirPath).listFiles();
		int i = 0;
		ClassifierContext classifier = setParams(model);
		for (File s : inputDir) {
			String[] doc;
			try {
				doc = preprocessDocument(s);
				ClassifierResult classifier_result;
				classifier_result = classifier.classifyDocument(doc, "unknown");
				if (classifier_result.getLabel().equals("neg")) {
					i = i + 1;
					System.out.println(i);
				}
				File file = new File(props.getProperty("OpDirPath")
						+ classifier_result.getLabel() + "/" + s.getName());
				FileUtils
						.writeStringToFile(file, FileUtils.readFileToString(s));
				System.out.println(s.getName() + " Category: "
						+ classifier_result.getLabel() + " Score: "
						+ classifier_result.getScore());
			} catch (InvalidDatastoreException e) {
				// If DataStore cannot be initialized.
				e.printStackTrace();
			} catch (IOException e) {
				// if cannot process document
				e.printStackTrace();
			}
		}
	}

	public static void testSingleInstance(String modelPath) {
		try {
			File model = new File(modelPath);
			ClassifierContext classifier = setParams(model);
			ClassifierResult classifier_result;
			System.out.println("Enter a review sentence ");
			String s = in.nextLine();

			StandardAnalyzer analyzer = new StandardAnalyzer(Version.LUCENE_31);
			StringReader sr = new StringReader(s);
			TokenStream tokensteam = analyzer.tokenStream(null, sr);
			ArrayList<String> tokenlist = new ArrayList<String>();
			while (tokensteam.incrementToken()) {
				tokenlist.add(tokensteam.getAttribute(CharTermAttribute.class)
						.toString());
			}
			
			String[] doc = tokenlist.toArray(new String[tokenlist.size()]);
			classifier_result = classifier.classifyDocument(doc, "unknown");
            System.out.println(classifier_result.getLabel());
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InvalidDatastoreException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

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

	public static void initializeDefaults() throws Exception {
		String fileName = "settings.properties";
		InputStream inputStream = new FileInputStream(fileName);
		props.load(inputStream);
	}

	/**
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		initializeDefaults();
		String model_path = props.getProperty("ModelPath"); // "/home/developer/corpus/models/wego_tweet_model/";
		String inputDir = props.getProperty("IpDirPath"); // "/home/developer/dataset_rev/freshrevs/test/pos");
		System.out.println(model_path + " " + inputDir);
		// "/home/developer/dataset_rev/freshrevs/test/neg/";
		//testClassifier(model_path, inputDir);
		testSingleInstance(model_path);
		
	}
}