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
parser.add_argument("-w", "--warmstart", default=False, type=bool)
parser.add_argument("-o", "--output_filename", required=True) # e.g. california_earthquake/tweets.p
parser.add_argument("-l1", "--volunteer_label", required=True) # e.g. california_earthquake/volunteer_labels.p
parser.add_argument("-l2", "--crowdflower_label", required=True) # e.g. california_earthquake/crowdflower_labels.p
parser.add_argument("-t", "--time_out", default=5, type=float)
parser.add_argument("-r", "--retries_only", default=False, type=bool)
args = parser.parse_args()

def run_batch(api, tweet_ids, tweet_text, retry_ids):
	successful_ids, tweets = [], []
	try:
		tweets = api.statuses_lookup(tweet_ids)
	except:
		print "Batch call to twitter failed, adding %d tweets to retry queue" % len(tweet_ids)
		retry_ids.extend(tweet_ids)
		return tweet_text, list(set(retry_ids))

	for tweet in tweets:
		successful_ids.append(tweet.id_str)
		tweet_text[tweet.id_str] = tweet

	unsuccessful_ids = set(tweet_ids) - set(successful_ids)
	retry_ids.extend(list(unsuccessful_ids))
	print "\tSuccessfully collected %d tweets, need to retry %d" % (len(successful_ids), len(unsuccessful_ids))
	return tweet_text, list(set(retry_ids))


auths = [tweepy.OAuthHandler(app['APP_KEY'], app['APP_SECRET']) for app in app_keys]
for i, auth in enumerate(auths): auth.set_access_token(app_keys[i]['ACCESS_TOKEN'], app_keys[i]['ACCESS_TOKEN_SECRET'])
apis = [tweepy.API(auth, wait_on_rate_limit_notify=True, wait_on_rate_limit=True, timeout=1) for auth in auths]

start_idx = args.start_idx
num_apis = len(apis)

assert(len(glob.glob(args.volunteer_label)) > 0), "Volunteer labels file %s invalid" % args.volunteer_label
assert(len(glob.glob(args.crowdflower_label)) > 0), "Crowdflower labels file %s invalid" % args.crowdflower_label

# Construct labels, potentially overwriting some of the volunteer with paid crowdflower labels
labels = {}
if args.volunteer_label:
	with open(args.volunteer_label, 'rb') as v_label_file:
		v_labels = pickle.load(v_label_file)
		for key in v_labels: labels[key] = v_labels[key]
if args.crowdflower_label:
	with open(args.crowdflower_label, 'rb') as c_label_file:
		c_labels = pickle.load(c_label_file)
		for key in c_labels: labels[key] = c_labels[key]

print "Found %d labels..." % len(labels)

# Iterate through all the tweets
num_tweets = 0
tweet_text = {}
print "Saving output at ", args.output_filename

tweet_ids = labels.keys()
total_tweets = len(tweet_ids)
retry_ids = []

# Warmstart if necessary
if (start_idx > 0 or args.warmstart or args.retries_only) and len(glob.glob(args.output_filename)) > 0:
	print "Warmstarting...."
	with open(args.output_filename, "rb") as input_file:
		tweet_text, retry_ids = pickle.load(input_file)
		print "Loaded %d tweets from previous run with %d retry ids" % (len(tweet_text), len(retry_ids))
	
	if start_idx == 0: start_idx = (len(tweet_text) + len(retry_ids))
	print "Starting from entry %d" % (start_idx)

# Iterate through each batch of tweet ids
for i in range(int(start_idx)/MAX_QUERIES, total_tweets/MAX_QUERIES):
	# Save all progress to the output file if we are at the checkpoint
	if i*MAX_QUERIES%args.checkpoint == 0:
		with open(args.output_filename, "wb") as output_file:
			pickle.dump([tweet_text, retry_ids], output_file)
			print "Saving %d tweets and %d retry ids..." % (len(tweet_text), len(retry_ids))
	tweet_text, retry_ids = run_batch(apis[i%num_apis], tweet_ids[i*MAX_QUERIES:(i+1)*MAX_QUERIES], tweet_text, retry_ids)
	time.sleep(args.time_out)
	print "Collected %d out of %d..." % (len(tweet_text), total_tweets)

# Iterate through remainders (if any)
if total_tweets % MAX_QUERIES > 0:
	tweet_text, retry_ids = run_batch(apis[0], tweet_ids[(total_tweets/MAX_QUERIES+1)*MAX_QUERIES:], tweet_text, retry_ids)
	print "Collected %d out of %d..." % (len(tweet_text), total_tweets)

# Iterate through each batch of retry_ids
total_retry_tweets = len(retry_ids)
print "Re-trying %d tweets which failed on the last round" % total_retry_tweets
for i in range(total_retry_tweets/MAX_QUERIES):
	if i*MAX_QUERIES%args.checkpoint == 0:
		with open(args.output_filename, "wb") as output_file:
			pickle.dump([tweet_text, retry_ids], output_file)
			print "Saving %d tweets and %d retry ids..." % (len(tweet_text), len(retry_ids))
	failed_retry_ids = []
	batch_retry_ids = retry_ids[i*MAX_QUERIES:(i+1)*MAX_QUERIES]
	tweet_text, failed_retry_ids = run_batch(apis[i%num_apis], batch_retry_ids, tweet_text, failed_retry_ids)
	successful_ids = set(batch_retry_ids) - set(failed_retry_ids)
	retry_ids = list(set(retry_ids)-successful_ids)
	print "Collected %d out of %d..." % (len(tweet_text), total_tweets)
	time.sleep(args.time_out)
