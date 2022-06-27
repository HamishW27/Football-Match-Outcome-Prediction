import pipeline
import unittest


class TestLinks(unittest.TestCase):

    def test_download_data(self):
        db = pipeline.download_data('football')
        self.assertIsInstance(db, pipeline.pd.DataFrame)

    def test_join_lists(self):
        list1 = [['https://www.besoccer.com/match/go-ahead-eagles/\
            heerenveen/20222326', 1],
                 ['https://www.besoccer.com/match/\
                    rkc-waalwijk/az/20222327', 1]]
        list2 = [['https://www.besoccer.com/match/\
            ajax/nec/20222330', 1],
                 ['https://www.besoccer.com/match/\
                    sc-cambuur-leeuwarden/fc-groningen/20222331',
                  1]]
        merged_lists = pipeline.join_lists([list1, list2])
        self.assertIsInstance(merged_lists, list)


if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=0, exit=False)
