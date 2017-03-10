from app_tokens import app_keys
import argparse
import csv
import glob
import pickle
import time
import tweepy

MAX_QUERIES = 100

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--checkpoint", default=1000, type=int)
parser.add_argument("-d", "--debug_every", default=100, type=int)
parser.add_argument("-c", "--start_idx", default=0, type=int)
parser.add_argument("-w", "--warmstart", default=True, type=bool)
parser.add_argument("-i", "--input_glob", required=True) # e.g. california_earthquake/tweet_ids/*.csv
parser.add_argument("-o", "--output_dir", required=True) # e.g. california_earthquake/tweets
parser.add_argument("-f", "--file_idx", default=0, type=int)
parser.add_argument("-t", "--time_out", default=5, type=float)
parser.add_argument("-r", "--retries_only", default=False, type=bool)
args = parser.parse_args()

auths = [tweepy.OAuthHandler(app['APP_KEY'], app['APP_SECRET']) for app in app_keys]
for i, auth in enumerate(auths): auth.set_access_token(app_keys[i]['ACCESS_TOKEN'], app_keys[i]['ACCESS_TOKEN_SECRET'])
apis = [tweepy.API(auth, wait_on_rate_limit_notify=True, wait_on_rate_limit=True, timeout=1) for auth in auths]

csv_filenames = glob.glob(args.input_glob)
start_idx = args.start_idx
num_apis = len(apis)

def run_batch(api, tweet_ids, output_filename, tweet_text, retry_ids):
	successful_ids, tweets = [], []
	try:
		tweets = api.statuses_lookup(tweet_ids)
	except:
		print "Batch call to twitter failed, adding %d tweets to retry queue" % len(tweet_ids)
		retry_ids.extend(tweet_ids)
		return tweet_text, list(set(retry_ids))

	for tweet in tweets:
		successful_ids.append(tweet.id_str)
		tweet_text[tweet.id_str] = tweet.text

	unsuccessful_ids = set(tweet_ids) - set(successful_ids)
	retry_ids.extend(list(unsuccessful_ids))
	print "\tSuccessfully collected %d tweets, need to retry %d" % (len(successful_ids), len(unsuccessful_ids))
	return tweet_text, list(set(retry_ids))

print "Found %d input files" % len(csv_filenames)
for filename in csv_filenames[int(args.file_idx):]:
	with open(filename, 'rb') as csvfile:
		
		output_filename = filename.split('/')[-1]
		output_filename = args.output_dir + "/" + output_filename.split('.')[0] + ".p"
		num_tweets = 0
		tweet_text = {}
		print "Saving output at ", output_filename

		# Initialize objects and csv reader
		reader = csv.reader(csvfile, delimiter=",", quotechar="'")
		reader.next() # Skip first header line that delineates each line as tweet_id, user_id
		entries = list(reader)
		tweet_ids = [entry[0] for entry in entries]
		total_tweets = len(tweet_ids)
		retry_ids = []

		# Warmstart if necessary
		if (start_idx > 0 or args.warmstart or args.retries_only) and len(glob.glob(output_filename)) > 0:
			print "Warmstarting...."
			with open(output_filename, "rb") as input_file:
				tweet_text, retry_ids = pickle.load(input_file)
				print "Loaded %d tweets from previous run with %d retry ids" % (len(tweet_text), len(retry_ids))
			
			if start_idx == 0: start_idx = (len(tweet_text) + len(retry_ids))
			print "Starting from entry %d" % (start_idx)

		# Iterate through each batch of tweet ids
		for i in range(int(start_idx)/MAX_QUERIES, total_tweets/MAX_QUERIES):
			# Save all progress to the output file if we are at the checkpoint
			if i*MAX_QUERIES%args.checkpoint == 0:
				with open(output_filename, "wb") as output_file:
					pickle.dump([tweet_text, retry_ids], output_file)
					print "Saving %d tweets and %d retry ids..." % (len(tweet_text), len(retry_ids))
			tweet_text, retry_ids = run_batch(apis[i%num_apis], tweet_ids[i*MAX_QUERIES:(i+1)*MAX_QUERIES], output_filename, tweet_text, retry_ids)
			time.sleep(args.time_out)
			print "Collected %d out of %d..." % (len(tweet_text), total_tweets)

		# Iterate through remainders (if any)
		if total_tweets % MAX_QUERIES > 0:
			tweet_text, retry_ids = run_batch(apis[0], tweet_ids[(i+1)*MAX_QUERIES:], output_filename, tweet_text, retry_ids)
			print "Collected %d out of %d..." % (len(tweet_text), total_tweets)

		# Iterate through each batch of retry_ids
		total_retry_tweets = len(retry_ids)
		for i in range(total_retry_tweets/MAX_QUERIES):
			if i*MAX_QUERIES%args.checkpoint == 0:
				with open(output_filename, "wb") as output_file:
					pickle.dump([tweet_text, retry_ids], output_file)
					print "Saving %d tweets and %d retry ids..." % (len(tweet_text), len(retry_ids))
			failed_retry_ids = []
			batch_retry_ids = retry_ids[i*MAX_QUERIES:(i+1)*MAX_QUERIES]
			tweet_text, failed_retry_ids = run_batch(apis[i%num_apis], batch_retry_ids, output_filename, tweet_text, failed_retry_ids)
			successful_ids = set(batch_retry_ids) - set(failed_retry_ids)
			retry_ids = list(set(retry_ids)-successful_ids)
			print "Collected %d out of %d..." % (len(tweet_text), total_tweets)
			time.sleep(args.time_out)
