package android_unity_test.android_unity_test;

import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

import customcode.customcode_android.CustomCodeActivity;
import vuforiatest.testtest.UnityPlayerActivity;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button ccbtn = (Button)findViewById(R.id.ccbutton);
        ccbtn.setOnClickListener(new Button.OnClickListener(){
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MainActivity.this, CustomCodeActivity.class);
                startActivity(intent);
                }
        }
        );

        Button unitybtn = (Button)findViewById(R.id.unitybutton);
        unitybtn.setOnClickListener(new Button.OnClickListener(){
                                     @Override
                                     public void onClick(View v) {
                                         Intent intent = new Intent(MainActivity.this, UnityPlayerActivity.class);
                                         startActivity(intent);
                                     }
                                 }
        );
    }
}
